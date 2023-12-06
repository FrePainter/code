import os
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
import argparse
import soundfile as sf
from functools import partial
from multiprocessing import Pool
import glob
import torch
import random
from scipy.signal import resample_poly
import librosa
import numpy as np

def resampling(wav, output_dir):
    audio, sr = librosa.load(wav, sr=None)
    if sr != 24000:
        inp_audio = librosa.resample(audio, sr, 24000, res_type='kaiser_best')
    else:
        inp_audio = np.copy(audio)
    if sr != 48000:
        src_audio = resample_poly(audio, 48000, sr)
    else:
        src_audio = np.copy(audio)

    inp_audio = inp_audio / np.abs(inp_audio).max() * 0.95 * 32768
    src_audio = src_audio / np.abs(src_audio).max() * 0.95 * 32768
    basename = os.path.splitext(os.path.basename(wav))[0] + '.wav'

    sf.write(os.path.join(output_dir, 'inp', basename), inp_audio.astype('int16'), samplerate=24000)
    sf.write(os.path.join(output_dir, 'src', basename), src_audio.astype('int16'), samplerate=48000)
    return None



def main(file_path, output_dir, ext):
    if os.path.splitext(file_path)[1] != '':
        files = [file_path]
    else:
        files = glob.glob(os.path.join(file_path, f'**/*.{ext}'), recursive=True)

    os.makedirs(os.path.join(output_dir, 'inp'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'src'), exist_ok=True)

    preprocess = partial(resampling, output_dir=output_dir)
    with Pool(os.cpu_count()) as p:
        list(tqdm(p.imap(preprocess, files), total=len(files), desc="Preprocessing ..."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, required=True)
    parser.add_argument('-o', '--output_dir', default='./logs/results/samples')
    parser.add_argument('-e', '--ext', default='wav')
    args = parser.parse_args()
    print(args.ext)
    main(args.dataset_path, args.output_dir, args.ext)