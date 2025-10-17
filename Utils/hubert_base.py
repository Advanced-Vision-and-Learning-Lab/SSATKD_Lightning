import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel, HubertConfig


import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel, HubertConfig
import pdb

class HuBERTBaseForClassification(nn.Module):
    """
    Teacher wrapper around HuBERT-Base (768-d hidden).
    Expects raw waveform input of shape (B, T) at sample_rate_in (default 32k).
    Internally resamples to 16k for HuBERT.
    """
    def __init__(self,
                 num_classes: int,
                 use_pretrained: bool = True,
                 freeze_encoder: bool = False,
                 sample_rate_in: int = 32000):
        super().__init__()

        self.num_classes = num_classes
        self.sample_rate_in = sample_rate_in
        self.sample_rate_hubert = 16000

        # Resampler
        if self.sample_rate_in != self.sample_rate_hubert:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate_in, new_freq=self.sample_rate_hubert
            )
        else:
            self.resampler = nn.Identity()

        # Load HuBERT
        if use_pretrained:
            self.encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        else:
            cfg = HubertConfig()
            self.encoder = HubertModel(cfg)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = self.encoder.config.hidden_size  # 768 for base
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )

    def _resample(self, x):
        return self.resampler(x)

    def forward(self, wave, return_hidden_states: bool = False):
        if wave.dim() != 2:
            raise ValueError(f"Expected wave of shape (B, T), got {wave.shape}")
    
        wave16 = self._resample(wave)
        out = self.encoder(wave16, output_hidden_states=True)  # always get hidden_states
        feats = out.last_hidden_state           # (B, T', H)
        pooled = feats.mean(dim=1)              # (B, H)
        logits = self.head(pooled)              # (B, C)
    
        # hidden_states[0]=embeddings, [1]=layer1, [2]=layer2, ...
        hs = out.hidden_states
        layer2 = hs[2] if len(hs) > 2 else feats
    
        features = {
            "pooled": pooled,       # (B, H)
            "sequence": feats,      # (B, T', H)
            "layer2": layer2,       # (B, T', H)  <-- what you need
        }
        if return_hidden_states:
            features["hidden_states"] = hs
        # pdb.set_trace()
        features = features["layer2"]
        features = features.transpose(1,2)
        features = features.unsqueeze(1)
        
        return features, logits
