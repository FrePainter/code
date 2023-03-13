import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from mel_processing import  mel_spectrogram_torch
import os
import argparse
import glob
from tqdm import tqdm
import numpy as np
import librosa
from random import randint as rdint
import random

def main(args):
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    port = 60000 + rdint(0, 100)
    os.environ['MASTER_PORT'] = str(port)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args))

    if args.save_audio:
        make_filelist_vctk(args)
    else:
        make_filelist_libritts(args)


def run(rank, n_gpus, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(1234)
    torch.cuda.set_device(rank)

    dset = DLoader(args)
    d_sampler = torch.utils.data.distributed.DistributedSampler(dset,
                                                                num_replicas=n_gpus,
                                                                rank=rank,
                                                                shuffle=True)
    collate_fn = Collate()
    d_loader = DataLoader(dset, num_workers=16, shuffle=False,
                            batch_size=1, pin_memory=True,
                            drop_last=False, collate_fn=collate_fn, sampler=d_sampler)

    prep(rank, d_loader, args)



def make_filelist_vctk(args):
    spks = os.listdir(args.output_dir)
    spks_test = ['p270', 'p256', 'p363', 'p241', 'p300', 'p336', 'p253', 'p266']
    spks_train = [spk for spk in spks if spk not in spks_test]

    npz_train = []
    npz_test = []

    for spk in spks_train:
        npz_train += glob.glob(os.path.join(args.output_dir, spk, '*.npz'))
    for spk in spks_test:
        npz_train += glob.glob(os.path.join(args.output_dir, spk, '*.npz'))
    random.seed(1234)
    random.shuffle(npz_train)
    random.shuffle(npz_test)
    with open(os.path.join(args.output_dir, 'filelist_vctk_train.txt'), 'w') as f:
        for npz in npz_train:
            f.write('{}\n'.format(npz))
    with open(os.path.join(args.output_dir, 'filelist_vctk_text.txt'), 'w') as f:
        for npz in npz_test:
            f.write('{}\n'.format(npz))

def make_filelist_libritts(args):
    npy_train = glob.glob(os.path.join(args.output_dir, 'train/*/*.npy'))
    npy_test = glob.glob(os.path.join(args.output_dir, 'test/*/*.pny'))
    with open(os.path.join(args.output_dir, 'filelist_libritts_train.txt'), 'w') as f:
        for npy in npy_train:
            f.write('{}\n'.format(npy))
    with open(os.path.join(args.output_dir, 'filelist_libritts_text.txt'), 'w') as f:
        for npy in npy_test:
            f.write('{}\n'.format(npy))


class DLoader():
    def __init__(self, args):
        if args.save_audio:
            self.wavs = glob.glob(os.path.join(args.input_dir, 'wav48_silence_trimmed/**/*_mic1.flac'), recursive=True)
        else:
            self.wavs = glob.glob(os.path.join(args.input_dir, '**/*.wav'), recursive=True)
        self.args = args
        print('wav num: ', len(self.wavs))

    def __getitem__(self, index):
        audio, sr = librosa.load(self.wavs[index], sr='none')
        audio, _ = librosa.effects.trim(audio,
                                        top_db=20,
                                        frame_length=2048,
                                        hop_length=300)
        audio = librosa.resample(audio, sr, self.args.samplerate)
        audio = audio / np.max(np.abs(audio)) * 0.95
        basename = os.path.splitext(os.path.basename(self.wavs[index]))[0].replace('_mic1','')
        spk_name = basename.split('_')[0]
        if self.args.save_audio:
            name = os.path.join(self.args.output_dir, spk_name, basename)
        else:
            split = self.wavs[index].replace(self.args.input_dir).rstrip().split('/')[0].split('-')[0]
            name = os.path.join(self.args.output_dir, split, spk_name, basename)
        return torch.FloatTensor(audio), name

    def __len__(self):
        return len(self.wavs)


class Collate():
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch[0][0], batch[0][1]


def prep(rank, d_loader, args):
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(d_loader)):
            audio, name = batch

            mel = mel_spectrogram_torch(audio.cuda(rank, non_blocking=True).unsqueeze(0),
                                        n_fft=2048,
                                        num_mels=128,
                                        sampling_rate=24000,
                                        hop_size=300,
                                        win_size=1200,
                                        fmin=0, fmax=None).squeeze(0)

            os.makedirs(os.path.dirname(name), exist_ok=True)
            if args.save_audio:
                data = {
                    'mel': mel.cpu(),
                    'audio': audio.cpu()
                }
                np.savez(name, **data)
            else:
                np.save(name, mel.cpu())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, help='Directory of audio files')
    parser.add_argument('-o', '--output_dir', required=True, help='Directory to save preprocessed data')
    parser.add_argument('-s', '--sample_rate', default=24000, help='Target sampling rate')
    parser.add_argument('--save_audio', action='store_true', help='Saving audio for fine-tuning (True: VCTK, False: LibriTTS')
    parser.add_argument('--testset', action='store_true', help='Generation of testset')
    a = parser.parse_args()

    main(a)