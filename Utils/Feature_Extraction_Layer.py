import torch.nn as nn
import torch
import torch.nn.functional as F
from nnAudio import features
from Demo_Parameters import Parameters
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import numpy as np
from .pytorch_utils import do_mixup
import pdb

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)
class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        batch_size = batch_size * 2
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)
    def mixup_data(x, y, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        batch_size = x.size()[0]    
        index = torch.randperm(batch_size)
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, window_length, window_size, hop_size, mel_bins, fmin, fmax, classes_num,
                 hop_length, sample_rate=8000, RGB=False, downsampling_factor=2):
        super(Feature_Extraction_Layer, self).__init__()

        # Convert window and hop length to ms
        window_length /= 1000
        hop_length /= 1000
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.bn = nn.Sequential()

        if RGB:
            num_channels = 3
            MFCC_padding = nn.ZeroPad2d((3, 6, 16, 16))
        else:
            num_channels = 1
            MFCC_padding = nn.ZeroPad2d((1, 0, 4, 0))
        
        
        self.num_channels = num_channels
        self.input_feature = input_feature

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        # Return Mel Spectrogram that is 48 x 48
        self.Log_Mel_Spectrogram = nn.Sequential(self.spectrogram_extractor,
                                                self.logmel_extractor,
                                                Transpose(1, 3),
                                                self.bn,
                                                Transpose(1, 3))
        
        #self.downsample = nn.AvgPool2d(kernel_size=(1, downsampling_factor))

        # Return Mel Spectrogram that is 48 x 48
        self.Mel_Spectrogram = nn.Sequential(features.mel.MelSpectrogram(sample_rate, n_mels=40, win_length=int(window_length*sample_rate),
                                                                          hop_length=int(
                                                                              hop_length*sample_rate),
                                                                          n_fft=int(window_length*sample_rate), verbose=False),
                                              nn.ZeroPad2d((1, 4, 0, 4)))

        # Return MFCC that is 16 x 48 (TDNN models) or 48 x 48 (CNNs)
        self.MFCC = nn.Sequential(features.mel.MFCC(sr=sample_rate, n_mfcc=16,
                                                    n_fft=int(
                                                        window_length*sample_rate),
                                                    win_length=int(
                                                        window_length*sample_rate),
                                                    hop_length=int(
                                                        hop_length*sample_rate),
                                                    n_mels=48, center=False, verbose=False), MFCC_padding)

        # # Return STFT that is 48 x 48
        # self.STFT = nn.Sequential(features.STFT(sr=sample_rate, n_fft=int(window_length*sample_rate),
        #                                         hop_length=int(
        #                                             hop_length*sample_rate),
        #                                         win_length=int(
        #                                             window_length*sample_rate),
        #                                         output_format='Magnitude',
        #                                         freq_bins=48, verbose=False), nn.ZeroPad2d((1, 4, 0, 0)))
        # Return STFT that is 48 x 48
        self.STFT = nn.Sequential(features.STFT(sr=sample_rate, n_fft=int( 0.1*sample_rate),
                                                hop_length=int(
                                                    0.025*sample_rate),
                                                win_length=int(
                                                    0.1*sample_rate),
                                                output_format='Magnitude',
                                                freq_bins=64, verbose=False), nn.ZeroPad2d((1, 4, 0, 0)))
        # Return GFCC that is 64 x 48
        self.GFCC = nn.Sequential(features.Gammatonegram(sr=sample_rate,
                                                         hop_length=int(
                                                             hop_length*sample_rate),
                                                         n_fft=int(
                                                             window_length*sample_rate),
                                                         verbose=False, n_bins=64), nn.ZeroPad2d((1, 0, 0, 0)))

        # Return CQT that is 64 x 48
        self.CQT = nn.Sequential(features.CQT(sr=sample_rate,
                                              hop_length=int(hop_length*sample_rate), n_bins=48,
                                               verbose=False), nn.ZeroPad2d((0, 0, 0, 0)))

        # Return VQT that is 64 x 48
        self.VQT = nn.Sequential(features.VQT(sr=sample_rate, hop_length=int(hop_length*sample_rate),
                                              n_bins=48, earlydownsample=False, verbose=False), nn.ZeroPad2d((0, 5, 0, 0)))

        self.features = {'Log_Mel_Spectrogram':self.Log_Mel_Spectrogram,'Mel_Spectrogram': self.Mel_Spectrogram,
                         'MFCC': self.MFCC, 'STFT': self.STFT, 'GFCC': self.GFCC,
                         'CQT': self.CQT, 'VQT': self.VQT}

    def forward(self, x):
        # pdb.set_trace()
        x = self.features[self.input_feature](x)
        x = x.repeat(1, self.num_channels, 1, 1)
        if self.training:
            x = self.spec_augmenter(x)
            # mixup = Mixup(mixup_alpha=1.)
            # mixup_lambda = mixup.get_lambda(32)
            # x = do_mixup(x, mixup_lambda)
        # #pdb.set_trace()
        if torch.isnan(x).any():
            raise ValueError(f"NaN values found in signal from file {x}")
            pdb.set_trace()
        return x
