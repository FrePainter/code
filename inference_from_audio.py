import os
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import nn
from tqdm import tqdm
import utils

from data_utils import (
    InpaintAudioLoader,
    InpaintAudioCollate,
)
from models_ft import (
    SynthesizerTrn,

)
import argparse
from audio_mae import AudioMaskedAutoencoderViT
from functools import partial
import random
import soundfile as sf
from mel_processing import mel_spectrogram_torch
torch.backends.cudnn.benchmark = True
global_step = 0

def main(args):
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    port = 60000 + random.randint(0, 1000)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)


    hps = utils.get_hparams_from_file(os.path.join('./logs/finetune', args.model, 'config.json'))
    if args.step:
        hps.ckpt_file = os.path.join('./logs/finetune', args.model, 'G_{}.pth'.format(args.step))
        hps.step = args.step
    else:
        hps.ckpt_file = utils.latest_checkpoint_path(os.path.join('./logs/finetune', args.model), "G_*.pth")
        hps.step = os.path.basename(hps.ckpt_file).replace('G_', '').replace('.pth', '')

    hps.folder = os.path.join(args.model)
    hps.step = str(args.step)
    hps.dataset_path = args.dataset_path
    hps.output_dir = args.output_dir

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    collate_fn = InpaintAudioCollate(hps.data, return_name=True)

    eval_dataset = InpaintAudioLoader(hps.dataset_path, hps.output_dir, hps.data, return_name=True)
    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,
                                                              num_replicas=n_gpus,
                                                              rank=rank,
                                                              shuffle=False)
    eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
                             batch_size=1, pin_memory=True,
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


    net_g.patch_embed = pt_encoder.patch_embed
    net_g.pos_embed = pt_encoder.pos_embed
    net_g.cls_token = pt_encoder.cls_token
    net_g.encoder = pt_encoder.encoder
    net_g.pt_norm = pt_encoder.norm

    del pt_encoder

    print(hps.ckpt_file)
    _, _, _, epoch_str = utils.load_checkpoint(hps.ckpt_file, net_g, None)


    global_step = 0

    synthesize(rank, net_g, eval_loader, hps.data)


def synthesize(rank, generator, eval_loader, hps):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (audio, audio_lengths, names) in enumerate(tqdm(eval_loader)):
            audio_lengths = audio_lengths.cuda(rank)
            mel = mel_spectrogram_torch(audio.cuda(rank),
                                        n_fft=hps.filter_length,
                                        num_mels=hps.n_mel_channels,
                                        sampling_rate=hps.sampling_rate,
                                        hop_size=hps.hop_length,
                                        win_size=hps.win_length,
                                        fmin=hps.mel_fmin, fmax=hps.mel_fmax)


            mel_lengths = audio_lengths // hps.hop_length
            y_hat = generator.infer(mel, mel_lengths, max_len=1000).squeeze(1)
            y_hat_total = torch.cat(torch.chunk(y_hat, y_hat.size(0), dim=0), dim=1)
            y_hat_total = y_hat_total[0,:audio_lengths.sum()]

            y_hat_total = y_hat_total / torch.abs(y_hat_total).max() * 0.95 * 32768.0
            y_hat_total = y_hat_total.data.cpu().numpy().astype('int16')
            sf.write(names, y_hat_total, samplerate=hps.sampling_rate)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('-s', '--step', type=str, default=200000)
    parser.add_argument('-d', '--dataset_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', default='./logs/results/samples')
    args = parser.parse_args()
    main(args)