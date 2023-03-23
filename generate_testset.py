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
from scipy.signal import sosfiltfilt
from scipy.signal import cheby1
from scipy.signal import resample_poly
from functools import partial
from multiprocessing import Pool

def main(args):
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    port = 60000 + rdint(0, 1000)
    os.environ['MASTER_PORT'] = str(port)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, args))


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

    prep(rank, d_loader, args, args.samplerate)




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


class DLoader():
    def __init__(self, args):
        self.wavs = []
        spks_test = ['p270', 'p256', 'p363', 'p241', 'p300', 'p336', 'p253', 'p266']
        for spk in spks_test:
            self.wavs += glob.glob(os.path.join(args.input_dir, 'wav48_silence_trimmed', spk, '*_mic1.flac'))

        self.args = args
        print('wav num: ', len(self.wavs))

    def __getitem__(self, index):
        audio, sr = librosa.load(self.wavs[index], sr=None)
        audio, _ = librosa.effects.trim(audio,
                                        top_db=20,
                                        frame_length=2048,
                                        hop_length=300)
        if len(audio) >= 384 * 300 * 2:
            return None, None, None
        basename = os.path.splitext(os.path.basename(self.wavs[index]))[0].replace('_mic1','')
        spk_name = basename.split('_')[0]
        name = os.path.join(spk_name, basename)
        return audio, sr, name

    def __len__(self):
        return len(self.wavs)


class Collate():
    def __init__(self):
        pass

    def __call__(self, batch):
        return batch[0][0], batch[0][1], batch[0][2]


def prep(rank, d_loader, args, target_sr):
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(d_loader)):
            audio, orig_sr, name = batch
            if not name:
                continue
            sampling_rates = [int(x) for x in args.test_samplerates.split('|')]
            gt_audio = resample_poly(audio, 24000, orig_sr)
            gt_audio = gt_audio / np.max(np.abs(gt_audio)) * 0.95
            gt_mel = mel_spectrogram_torch(torch.FloatTensor(gt_audio).cuda(rank).unsqueeze(0),
                                        n_fft=2048,
                                        num_mels=128,
                                        sampling_rate=24000,
                                        hop_size=300,
                                        win_size=1200,
                                        fmin=0, fmax=None).squeeze(0)
            for sr in sampling_rates:
                # Degrading audio
                highcut = sr // 2
                nyq = target_sr // 2
                hi = highcut / nyq
                sos = cheby1(8, 0.05, hi, btype='lowpass', output='sos')
                d_audio = sosfiltfilt(sos, audio)
                d_audio = resample_poly(d_audio, highcut * 2, orig_sr)
                d_audio = resample_poly(d_audio, 24000, highcut * 2)
                d_audio = d_audio / np.max(np.abs(d_audio)) * 0.95
                d_mel = mel_spectrogram_torch(torch.FloatTensor(d_audio).cuda(rank).unsqueeze(0),
                                        n_fft=2048,
                                        num_mels=128,
                                        sampling_rate=24000,
                                        hop_size=300,
                                        win_size=1200,
                                        fmin=0, fmax=None).squeeze(0)

                data = {
                    'mel': d_mel.cpu(),
                    'audio': d_audio,
                    'mel_orig': gt_mel.cpu(),
                    'audio_orig': gt_audio
                }
                filename = os.path.join(args.output_dir, str(sr), name)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                np.savez(filename, **data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, help='Directory of audio files')
    parser.add_argument('-o', '--output_dir', required=True, help='Directory to save preprocessed data')
    parser.add_argument('-s', '--samplerate', default=24000, help='Target sampling rate')
    parser.add_argument('-t', '--test_samplerates', default='2000|4000|8000|12000|16000')
    a = parser.parse_args()

    main(a)