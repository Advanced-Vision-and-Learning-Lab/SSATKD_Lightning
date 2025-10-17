import os
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Config

class Wav2Vec2AudioEncoder(nn.Module):
    """
    Wav2Vec2-Base (audio-encoder-only) with LOCAL weight loading (no downloads).
    Expects raw waveform (B, T) at sample_rate_in (default 32k). Internally resamples to 16k.

    Behavior:
      - use_pretrained=True  -> build from config, then LOAD LOCAL CHECKPOINT into self.encoder
      - use_pretrained=False -> build from config (random init)
    """
    def __init__(self,
                 num_classes: int,
                 use_pretrained: bool = True,
                 freeze_encoder: bool = False,
                 sample_rate_in: int = 32000,
                 feature_source: str = "cnn",  # "cnn" | "last_hidden" | "layer2"
                 local_checkpoint_path: str = "./Model Weights/wav2vec2_base.pt",
                 strict_load: bool = False):
        super().__init__()
        assert feature_source in ("cnn", "last_hidden", "layer2")
        self.feature_source = feature_source
        self.sample_rate_in = sample_rate_in
        self.sample_rate_w2v = 16000

        # Resampler
        if self.sample_rate_in != self.sample_rate_w2v:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate_in, new_freq=self.sample_rate_w2v
            )
        else:
            self.resampler = nn.Identity()

        # Always initialize from config (no download)
        cfg = Wav2Vec2Config()
        cfg.output_hidden_states = True
        self.encoder = Wav2Vec2Model(cfg)

        # If requested, load LOCAL pretrained weights
        if use_pretrained:
            if not os.path.exists(local_checkpoint_path):
                raise FileNotFoundError(f"Local Wav2Vec2 weights not found at: {local_checkpoint_path}")

            state_dict = torch.load(local_checkpoint_path, map_location="cpu")

            # Unwrap nested dicts if needed
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # (Optional but helpful) normalize common prefixes seen in checkpoints
            def strip_prefix(d, prefix):
                return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in d.items() }

            state_dict = strip_prefix(state_dict, "model.encoder.")
            state_dict = strip_prefix(state_dict, "encoder.")
            state_dict = strip_prefix(state_dict, "wav2vec2.")

            missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
            if strict_load and (missing or unexpected):
                raise RuntimeError(f"Strict load failed. missing={missing}, unexpected={unexpected}")
            print(f"[Wav2Vec2] Loaded local weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")

        

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = self.encoder.config.hidden_size  # 768
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )
        for p in self.head.parameters():
            p.requires_grad = True

        # Optional adapters if you want to force a fixed channel count for feats_4d
        self.to_fixed_channels = None  # e.g., nn.LazyConv2d(16, 1) if needed later

    @torch.no_grad()
    def _canon_wave(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B,T) or (B,1,T) or (B,C,T)->mixdown mono
        if x.dim() == 2:
            wave = x
        elif x.dim() == 3:
            if x.shape[1] != 1:
                x = x.mean(dim=1, keepdim=True)
            wave = x[:, 0, :]
        else:
            raise ValueError(f"Expected (B,T) or (B,*,T), got {x.shape}")
        return wave

    @torch.no_grad()
    def _resample(self, wave: torch.Tensor) -> torch.Tensor:
        return self.resampler(wave)

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor | None = None):
        """
        wave: (B,T) or (B,1,T) raw waveform in [-1,1]
        lengths: optional (B,) valid lengths at sample_rate_in
        returns: (feats_4d, logits)
        """
        # Canonicalize & resample
        wave = self._canon_wave(wave)       # (B,T)
        wave16 = self._resample(wave)       # (B,T16)

        # Attention mask (# Create a 16 kHz-scaled attention mask indicating valid (non-padded) audio regions.)
        attention_mask = None
        if lengths is not None:
            ratio = self.sample_rate_w2v / float(self.sample_rate_in)
            L16 = (lengths.float() * ratio).round().long()
            max_len = wave16.shape[1]
            attention_mask = (torch.arange(max_len, device=wave16.device)[None, :] < L16[:, None]).to(torch.long)

        # Encoder forward
        out = self.encoder(input_values=wave16, attention_mask=attention_mask)

        # Sequence features for logits
        seq = out.last_hidden_state            # (B, T', H)
        pooled = seq.mean(dim=1)               # (B, H)
        logits = self.head(pooled)             # (B, C)

        # Choose feature source for 4D map
        if self.feature_source == "cnn":
            # CNN (feature extractor) features:
            # out.extract_features: (B, T_c, C_c) in HF
            cnn_seq = out.extract_features     # (B, T_c, C_c)
            feats_4d = cnn_seq.transpose(1, 2).unsqueeze(-1)  # (B, C_c, T_c, 1)

        # Optional channel adapter
        if self.to_fixed_channels is not None:
            feats_4d = self.to_fixed_channels(feats_4d)
        
        feats_4d = feats_4d.squeeze(3).unsqueeze(1)
        return feats_4d, logits
