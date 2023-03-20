import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import commons
import utils
from data_utils import (
    MelLoader,
    MelCollate,
    DistributedBucketSampler
)
from audio_mae import (
    AudioMaskedAutoencoderViT
)
from functools import partial
import random

torch.backends.cudnn.benchmark = True
global_step = 0

def main():
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    port = 60000 + random.randint(0, 1000)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    hps = utils.get_hparams_pt()
    print("Masking ratio: ", hps.mask_ratio)
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = MelLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32,300,400,500,600,700,800, 832],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = MelCollate(hps.data)
    train_loader = DataLoader(train_dataset, num_workers=16, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=train_sampler)

    if rank == 0:
        eval_dataset = MelLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=16, shuffle=False,
                                 batch_size=hps.train.batch_size, pin_memory=True,
                                 drop_last=False, collate_fn=collate_fn)
    hm = hps.model_MAE
    net_g = AudioMaskedAutoencoderViT(
        hm.num_mels, hm.mel_len, hm.patch_size, hm.in_chans,
        hm.embed_dim, hm.encoder_depth, hm.num_heads,
        hm.decoder_embed_dim, hm.decoder_depth, hm.decoder_num_heads,
        hm.mlp_ratio, hm.mask_token_valu, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    net_g = DDP(net_g, device_ids=[rank])


    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank==0:
            train_and_evaluate(rank, epoch, hps, [net_g], [optim_g], [scheduler_g], scaler, [train_loader, eval_loader], logger, [writer, writer_eval], hps.mask_ratio)
        else:
            train_and_evaluate(rank, epoch, hps, [net_g], [optim_g], [scheduler_g], scaler, [train_loader, None], None, None, hps.mask_ratio)
        scheduler_g.step()

        if epoch % (hps.train.epochs - 1) == 0:
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
        if epoch % (hps.train.epochs) == 0:
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, mask_ratio):
    net_g = nets[0]
    optim_g = optims[0]
    # scheduler_g = schedulers[0]
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    for batch_idx, (mel, mel_lengths) in enumerate(train_loader):
        mel, mel_lengths = mel.cuda(rank, non_blocking=True), mel_lengths.cuda(rank, non_blocking=True)


        with autocast(enabled=hps.train.fp16_run):
            loss_mae = net_g(mel, mel_lengths, mask_ratio)

            with autocast(enabled=False):
                loss_gen_all = loss_mae

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_mae]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {"learning_rate": lr, "loss/g/train": loss_gen_all, "grad_norm_g": grad_norm_g, "loss/g/all": loss_gen_all}



                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, mask_ratio)
            if global_step % hps.train.save_interval == 0:
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))


        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))




def evaluate(hps, generator, eval_loader, writer_eval, mask_ratio):
    generator.eval()
    loss_tot = 0

    val_nums = 0
    with torch.no_grad():
        for batch_idx, (mel, mel_lengths) in enumerate(eval_loader):
            mel, mel_lengths = mel.cuda(0), mel_lengths.cuda(0)

            loss, _ = generator.module.infer(mel, mel_lengths, mask_ratio)

            loss_tot += loss * mel.size(0)

            val_nums += mel.size(0)

            # remove else
            mel = mel[:4]
            mel_lengths = mel_lengths[:4]


            if batch_idx >= 4:
                break

        loss_tot /= val_nums

        _, y_hat_mel = generator.module.infer(mel, mel_lengths, mask_ratio)

    image_dict = {
        "gen/mel1": utils.plot_spectrogram_to_numpy(y_hat_mel[0, :, :mel_lengths[0]].cpu().numpy()),
        "gen/mel2": utils.plot_spectrogram_to_numpy(y_hat_mel[1, :, :mel_lengths[1]].cpu().numpy()),
        "gen/mel3": utils.plot_spectrogram_to_numpy(y_hat_mel[2, :, :mel_lengths[2]].cpu().numpy()),
        "gen/mel4": utils.plot_spectrogram_to_numpy(y_hat_mel[3, :, :mel_lengths[3]].cpu().numpy())
    }

    scalar_dict = {'loss/g/val': loss_tot, "loss/g/all": loss_tot}
    if global_step == 0:
        image_dict.update({"gt/mel1": utils.plot_spectrogram_to_numpy(mel[0, :, :mel_lengths[0]].cpu().numpy())})
        image_dict.update({"gt/mel2": utils.plot_spectrogram_to_numpy(mel[1, :, :mel_lengths[1]].cpu().numpy())})
        image_dict.update({"gt/mel3": utils.plot_spectrogram_to_numpy(mel[2, :, :mel_lengths[2]].cpu().numpy())})
        image_dict.update({"gt/mel4": utils.plot_spectrogram_to_numpy(mel[3, :, :mel_lengths[3]].cpu().numpy())})


    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audio_sampling_rate=hps.data.sampling_rate,
        scalars=scalar_dict
    )
    generator.train()



if __name__ == "__main__":
    main()
