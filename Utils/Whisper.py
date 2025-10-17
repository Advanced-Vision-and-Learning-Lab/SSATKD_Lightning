import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig
import pdb

class WhisperEncoderFromSpec(nn.Module):
    """
    Whisper Base (encoder-only) using OUR mel features.
    Inputs:
      - x: (B, 1, T_frames, F) or (B, F, T_frames)  [your spectrogram]
         OR (B, T) / (B, 1, T) if you also pass a front_end that maps raw->(B, 80, T_frames).
    Returns:
      - feats_4d: (B, C, T_frames, 1)   # 2D-friendly for the texture heads
      - logits : (B, num_classes)
    """
    def __init__(self,
                 num_classes: int,
                 model_name: str = "openai/whisper-base",
                 use_pretrained: bool = True,
                 freeze_encoder: bool = False,
                 feature_source: str = "layer2",   # "layer2" | "last_hidden"
                 front_end: nn.Module | None = None # optional: your raw->mel
                 ):
        super().__init__()
        assert feature_source in ("layer2", "last_hidden")
        self.feature_source = feature_source
        self.front_end = front_end   # if provided, will be used for raw input

        # Build encoder-only with hidden states
        if use_pretrained:
            cfg = WhisperConfig.from_pretrained(model_name)
            cfg.output_hidden_states = True
            self.encoder = WhisperModel.from_pretrained(model_name, config=cfg).encoder
        else:
            cfg = WhisperConfig()
            cfg.output_hidden_states = True
            self.encoder = WhisperModel(cfg).encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = self.encoder.config.d_model  # 512 for whisper-base
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )

        # If your mel feature bins != 80, adapt channels with a 1x1 conv (Lazy infers in_channels).
        self.to80 = nn.LazyConv1d(80, kernel_size=1)  # applied on (B, F, T)

        # Optional: force a fixed channels C for feats_4d (not needed usually)
        self.to_fixed_channels = None  # e.g., nn.LazyConv2d(16, kernel_size=1)



    def _canon_input_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept:
          - (B, 1, T, F) -> (B, F, T)
          - (B, F, T)    -> (B, F, T)
          - (B, T) / (B,1,T) -> requires self.front_end
        Output:
          - (B, 80, T') for Whisper encoder
        """
        # raw -> require front_end
        if x.dim() == 2 or (x.dim() == 3 and x.shape[1] in (1,)):
            if self.front_end is None:
                raise ValueError("Raw input given but no front_end provided (raw -> (B,80,T')).")
            feats = self.front_end(x)  # expect (B, 80, T') or (B,1,T',80)
            if feats.dim() == 4:       # (B,1,T',80)
                feats = feats.squeeze(1).transpose(1, 2)  # -> (B,80,T')
            return feats
    
        # spectrogram paths
        if x.dim() == 4 and x.shape[1] == 1:   # (B,1,T,F)
            x = x.squeeze(1).transpose(1, 2)   # -> (B,F,T)
        elif x.dim() != 3:
            raise ValueError(f"Unexpected input shape {x.shape}. Expect (B,1,T,F) or (B,F,T) or raw with front_end.")
    
        B, Freq, T = x.shape
        if Freq != 80:
            # Param-free resize to 80 mel bins
            # (B,F,T) -> (B,1,F,T) -> interpolate F->80 -> (B,80,T)
            x = F.interpolate(x.unsqueeze(1), size=(80, T), mode='bilinear', align_corners=False).squeeze(1)
    
        return x  # (B,80,T)


    def forward(self, x: torch.Tensor):
        """
        x: your spectrogram (preferred) or raw (if front_end provided)
        Returns: (feats_4d, logits)
        """
        input_features = self._canon_input_features(x)       # (B, 80, T)
        T = input_features.shape[-1]
        if T < 3000:
            input_features = F.pad(input_features, (0, 3000 - T))        # pad right along time
        elif T > 3000:
            input_features = input_features[..., :3000]
        # pdb.set_trace()
        out = self.encoder(input_features=input_features)    # last_hidden: (B, T, 512)

        seq = out.last_hidden_state                          # (B, T, H)
        pooled = seq.mean(dim=1)                             # (B, H)
        logits = self.head(pooled)                           # (B, num_classes)

        # Choose feature source and convert to 4D (B, C, T, 1)
        if self.feature_source == "layer2" and out.hidden_states is not None and len(out.hidden_states) > 2:
            layer2 = out.hidden_states[2]                    # (B, T, H)
            feats_4d = layer2.transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
        else:
            feats_4d = seq.transpose(1, 2).unsqueeze(-1)     # (B, H, T, 1)

        if self.to_fixed_channels is not None:
            feats_4d = self.to_fixed_channels(feats_4d)

        return feats_4d.squeeze(3).unsqueeze(1), logits
