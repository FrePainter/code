import os
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
import utils

from data_utils import (
    InferenceTestLoader,
    InferenceTestCollate,
)
from models_ft import (
    SynthesizerTrn,

)
import argparse
from audio_mae import AudioMaskedAutoencoderViT
from functools import partial
import random
import soundfile as sf
from post_processing import PostProcessing

torch.backends.cudnn.benchmark = True
global_step = 0

def main(args):
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    port = 60000 + random.randint(0, 1000)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)


    hps = utils.get_hparams_from_file(os.path.join('./logs/finetune', args.model, 'config.json'))

    if args.step == 'best':
        hps.ckpt_file = os.path.join('./logs/finetune', args.model, 'best.pth')
        hps.step = 'best'
    elif args.step:
        hps.ckpt_file = os.path.join('./logs/finetune', args.model, 'G_{}.pth'.format(args.step))
        hps.step = args.step
    elif os.path.isfile(os.path.join('./logs/finetune', args.model, 'best.pth')):
        hps.ckpt_file = os.path.join('./logs/finetune', args.model, 'best.pth')
        hps.step = 'best'
    else:
        hps.ckpt_file = utils.latest_checkpoint_path(os.path.join('./logs/finetune', args.model), "G_*.pth")
        hps.step = os.path.basename(hps.ckpt_file).replace('G_', '').replace('.pth', '')

    hps.folder = os.path.join(args.model)
    hps.step = str(hps.step)
    hps.dataset_path = args.dataset_path

    args.output_dir = os.path.join('./logs/results', '{}_{}'.format(args.model, hps.step))

    print(f'Target sampling rate: {hps.data.sampling_rate} Hz')
    print(f'CKPT file:', hps.ckpt_file)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps, args))


def run(rank, n_gpus, hps, args):
    global global_step

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    collate_fn = InferenceTestCollate(hps.data)

    eval_dataset = InferenceTestLoader(args, rank)
    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,
                                                              num_replicas=n_gpus,
                                                              rank=rank,
                                                              shuffle=True)
    eval_loader = DataLoader(eval_dataset, num_workers=16, shuffle=False, batch_size=1,
                             drop_last=False, collate_fn=collate_fn, sampler=sampler)
    hm = hps.model_MAE
    pt_encoder = AudioMaskedAutoencoderViT(hm.num_mels, hm.mel_len, hm.patch_size, hm.in_chans,
                                           hm.embed_dim, hm.encoder_depth, hm.num_heads,
                                           hm.decoder_embed_dim, hm.decoder_depth, hm.decoder_num_heads,
                                           hm.mlp_ratio, hm.mask_token_valu, norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                           ).cuda(rank)

    net_g = SynthesizerTrn(
        hps.data.n_mel_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mask_ratio=0,
        **hps.model).cuda(rank)

    utils.joint_model(net_g, pt_encoder)
    del pt_encoder
    utils.load_checkpoint(hps.ckpt_file, net_g, None)

    pp = PostProcessing(rank)
    synthesize(rank, net_g, eval_loader, int(hps.data.sampling_rate), pp)


def synthesize(rank, generator, eval_loader, sampling_rate, pp):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (mel, mel_lengths, src_audio, names) in enumerate(tqdm(eval_loader)):
            mel, mel_lengths = mel.cuda(rank), mel_lengths.cuda(rank)
            src_audio = src_audio.cuda(rank)
            audio_lengths = mel_lengths * sampling_rate / 80

            y_hat = generator.infer(mel, mel_lengths).squeeze(1)

            y_hat_total = torch.cat(torch.chunk(y_hat, y_hat.size(0), dim=0), dim=1)
            y_hat_total = y_hat_total[:, :audio_lengths.sum().long()]

            y_hat_total = y_hat_total / torch.abs(y_hat_total).max() * 0.95

            y_hat_pp = pp.post_processing(y_hat_total, src_audio.squeeze(0), length=src_audio.size(-1))

            y_hat_pp = (y_hat_pp * 32768).squeeze().cpu().numpy().astype('int16')
            os.makedirs(os.path.dirname(names), exist_ok=True)
            sf.write(names, y_hat_pp, samplerate=sampling_rate)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('-s', '--step', type=str, default=None)
    parser.add_argument('-d', '--dataset_path', type=str, required=True)
    args = parser.parse_args()
    main(args)