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
import soundfile as sf

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



class DLoader():
    def __init__(self, args):
        wavs = glob.glob(os.path.join(args.input_dir, '**/*_mic1.flac'), recursive=True)
        spk_list = sorted(list(set([os.path.basename(wav).split('_')[0] for wav in wavs])))[-8:]

        self.wavs = []
        for wav in wavs:
            spk = os.path.basename(wav).split('_')[0]
            if spk in spk_list:
                self.wavs.append(wav)

        self.args = args
        print('wav num: ', len(self.wavs))

    def __getitem__(self, index):
        audio, sr = librosa.load(self.wavs[index], sr=None)
        audio, _ = librosa.effects.trim(audio,
                                        top_db=20,
                                        frame_length=2048,
                                        hop_length=600)

        if len(audio) % 600 != 0:
            audio = np.pad(audio, (0,600 - (len(audio)%600)), 'constant', constant_values=0)

        basename = os.path.splitext(os.path.basename(self.wavs[index]))[0].replace('_mic1','')
        return audio, sr, basename

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
            audio, orig_sr, basename = batch
            sampling_rates = [int(x) for x in args.test_samplerates.split('|')]

            for sr in sampling_rates:
                # Degrading audio
                highcut = sr // 2
                nyq = orig_sr // 2
                hi = highcut / nyq

                sos = cheby1(8, 0.05, hi, btype='lowpass', output='sos')
                d_audio = sosfiltfilt(sos, audio)
                d_audio = librosa.resample(d_audio, orig_sr, sr, res_type='kaiser_best')
                d_audio = librosa.resample(d_audio, sr, target_sr, res_type='kaiser_best')
                d_audio = d_audio / np.max(np.abs(d_audio)) * 0.95

                if len(d_audio) * 2 > len(audio):
                    d_audio = d_audio[:len(audio)]
                elif len(d_audio) * 2 < len(audio):
                    d_audio = np.pad(d_audio, (0, len(audio) - len(d_audio) * 2), 'constant', constant_values=0)

                d_mel = mel_spectrogram_torch(torch.FloatTensor(d_audio).cuda(rank).unsqueeze(0),
                                        n_fft=2048,
                                        num_mels=128,
                                        sampling_rate=24000,
                                        hop_size=300,
                                        win_size=1200,
                                        fmin=20, fmax=12000).squeeze(0)


                d_audio = (d_audio * 32768).astype('int16')


                src_wav = os.path.join(args.output_dir, 'source_wavs', str(sr), basename+'.wav')
                gt_wav = os.path.join(args.output_dir, 'gt_wavs', basename+'.wav')
                src_mel = os.path.join(args.output_dir, 'source_mels', str(sr), basename+'.pt')

                for file_ in [src_wav, gt_wav, src_mel]:
                    os.makedirs(os.path.dirname(file_), exist_ok=True)

                sf.write(src_wav, d_audio, samplerate=24000)
                torch.save(d_mel.cpu().squeeze(), src_mel)
            audio = (audio / np.max(np.abs(audio)) * 0.95 * 32768).astype('int16')
            sf.write(gt_wav, audio, samplerate=48000)










if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', required=True, help='Directory of audio files')
    parser.add_argument('-o', '--output_dir', required=True, help='Directory to save preprocessed data')
    parser.add_argument('-s', '--samplerate', default=24000, help='Target sampling rate')
    parser.add_argument('-t', '--test_samplerates', default='2000|4000|8000|12000|16000|24000')
    a = parser.parse_args()



    main(a)