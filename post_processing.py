from torchaudio.transforms import Spectrogram, InverseSpectrogram
import torch

class PostProcessing:
    def __init__(self, rank):
        self.stft = Spectrogram(2048, hop_length=512, win_length=2048, power=None, pad_mode='constant').cuda(rank)
        self.istft = InverseSpectrogram(2048, hop_length=512, win_length=2048, pad_mode='constant').cuda(rank)

    def get_cutoff_index(self, spec, threshold=0.985):
        energy = torch.cumsum(torch.sum(spec.squeeze().abs(), dim=-1), dim=0)
        threshold = energy[-1] * threshold
        for i in range(1, energy.size(0)):
            if energy[-i] < threshold:
                return energy.size(0) - i
        return 0
    def post_processing(self, pred, src, length):
        assert len(pred.shape) == 2 and len(src.shape) == 2
        spec_pred = self.stft(pred)
        spec_src  = self.stft(src)

        cr = self.get_cutoff_index(spec_src)
        energy_ratio = torch.mean(
            spec_src[:, cr].abs().sum() /
            spec_pred[:, cr, ...].abs().sum()
        )
        energy_ratio = min(max(energy_ratio, 0.8), 1.2)
        spec_pred[:, :cr, ...] = spec_src[:, :cr, ...] / energy_ratio

        audio = self.istft(spec_pred, length=length)
        audio = audio / torch.abs(audio).max() * 0.95

        return audio