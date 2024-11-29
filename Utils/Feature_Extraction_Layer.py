import torch.nn as nn
import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, window_length, window_size, hop_size, mel_bins, fmin, fmax, classes_num,
                 hop_length, sample_rate=8000, RGB=False, downsampling_factor=2, frame_shift=10.0):
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
        self.frame_shift = frame_shift  
        self.bn = nn.BatchNorm2d(64)

        
        if RGB:
            num_channels = 3
        else:
            num_channels = 1
        
        
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
        self.spec_augmenter = SpecAugmentation(time_drop_width=48, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        # Return Mel Spectrogram that is 48 x 48
        self.Log_Mel_Spectrogram = nn.Sequential(self.spectrogram_extractor,
                                                self.logmel_extractor,
                                                Transpose(1, 3),
                                                self.bn,
                                                Transpose(1, 3))


        self.features = {'Log_Mel_Spectrogram':self.Log_Mel_Spectrogram}

    
    def forward(self, x):
        x = self.features[self.input_feature](x)
        x = x.repeat(1, self.num_channels, 1, 1)
        if self.training:
            x = self.spec_augmenter(x)
        return x
    