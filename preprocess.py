import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from mel_processing import mel_spectrogram_torch
import os
import argparse
import glob
from tqdm import tqdm
import numpy as np
import librosa
from random import randint as rdint
import random
from functools import partial
from multiprocessing import Pool

def main(args):
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    port = 60000 + rdint(0, 1000)
    os.environ['MASTER_PORT'] = str(port)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args))
    os.makedirs('./filelists', exist_ok=True)
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

def mp_npy(npy):
    length = np.load(npy).shape[1]
    if length >=64 and length <=1000:
        return [npy, length]
def mp_npz(npz):
    length = np.load(npz)['mel'].shape[1]
    if length >=32 and length <=1000:
        return [npz, length]




def make_filelist_vctk(args):
    spks = os.listdir(args.output_dir)
    spks_test = ['p270', 'p256', 'p363', 'p241', 'p300', 'p336', 'p253', 'p266']
    spks_train = [spk for spk in spks if spk not in spks_test]

    npz_train = []
    npz_test = []

    for spk in spks_train:
        npz_train += glob.glob(os.path.join(args.output_dir, spk, '*.npz'))
    for spk in spks_test:
        npz_test += glob.glob(os.path.join(args.output_dir, spk, '*.npz'))

    mp_func = partial(mp_npz)
    with Pool(40) as p:
        rets = list(tqdm(p.imap(mp_func, npz_train), total=len(npz_train)))
        rets = list(filter(None, rets))
        rets.sort()
    with open('./filelists/filelist_vctk_train.txt', 'w') as f:
        for ret in rets:
            f.write('{}|{}\n'.format(ret[0], ret[1]))

    with Pool(40) as p:
        rets = list(tqdm(p.imap(mp_func, npz_test), total=len(npz_test)))
        rets = list(filter(None, rets))
        rets.sort()
    with open('./filelists/filelist_vctk_test.txt', 'w') as f:
        for ret in rets:
            f.write('{}|{}\n'.format(ret[0], ret[1]))

def make_filelist_libritts(args):
    npy_train = glob.glob(os.path.join(args.output_dir, 'train/*/*.npy'))
    npy_test = glob.glob(os.path.join(args.output_dir, 'test/*/*.npy'))

    mp_func = partial(mp_npy)
    with Pool(40) as p:
        rets = list(tqdm(p.imap(mp_func, npy_train), total=len(npy_train)))
        rets = list(filter(None, rets))
        rets.sort()
    with open('./filelists/filelist_libritts_train.txt', 'w') as f:
        for ret in rets:
            f.write('{}|{}\n'.format(ret[0], ret[1]))

    with Pool(40) as p:
        rets = list(tqdm(p.imap(mp_func, npy_test), total=len(npy_test)))
        rets = list(filter(None, rets))
        rets.sort()
        random.seed(1234)
        rets = random.sample(rets, 500)
    with open('./filelists/filelist_libritts_test.txt', 'w') as f:
        for ret in rets:
            f.write('{}|{}\n'.format(ret[0], ret[1]))


class DLoader():
    def __init__(self, args):
        if args.save_audio:
            self.wavs = glob.glob(os.path.join(args.input_dir, 'wav48_silence_trimmed/**/*_mic1.flac'), recursive=True)
        else:
            self.wavs = glob.glob(os.path.join(args.input_dir, '**/*.wav'), recursive=True)
        self.args = args
        print('wav num: ', len(self.wavs))

    def __getitem__(self, index):
        audio, sr = librosa.load(self.wavs[index], sr=None)
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
            split = self.wavs[index].replace(self.args.input_dir,'').strip('/').split('/')[0].split('-')[0]
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
    parser.add_argument('-s', '--samplerate', default=24000, help='Target sampling rate')
    parser.add_argument('--save_audio', action='store_true', help='Saving audio for fine-tuning (True: VCTK, False: LibriTTS')
    a = parser.parse_args()

    main(a)