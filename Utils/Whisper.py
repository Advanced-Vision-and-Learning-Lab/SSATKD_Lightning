import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperConfig

class WhisperEncoderFromSpec(nn.Module):
    """
    Whisper Base (encoder-only) using OUR mel features.
    Loads weights from a LOCAL checkpoint when use_pretrained=True.
    Inputs:
      - x: (B, 1, T, F) or (B, F, T)  [your spectrogram]
         OR (B, T) / (B, 1, T) if you also pass a front_end that maps raw->(B, 80, T).
    Returns:
      - feats_4d: (B, C, T, 1)   # 2D-friendly for the texture heads
      - logits : (B, num_classes)
    """
    def __init__(self,
                 num_classes: int,
                 model_name: str = "openai/whisper-base",   # kept for API symmetry; not used to download
                 use_pretrained: bool = True,
                 freeze_encoder: bool = False,
                 feature_source: str = "layer2",   # "layer2" | "last_hidden"
                 front_end: nn.Module | None = None,  # optional: your raw->mel
                 local_checkpoint_path: str | None = "./Model Weights/whisper_tiny.pt",
                 strict_load: bool = False):
        super().__init__()
        assert feature_source in ("layer2", "last_hidden")
        self.feature_source = feature_source
        self.front_end = front_end

        # Build full Whisper from config (NO download), then take encoder submodule
        cfg = WhisperConfig()
        cfg.output_hidden_states = True
        full = WhisperModel(cfg)
        self.encoder = full.encoder  # we only keep the encoder

        # Load LOCAL weights into encoder if requested
        if use_pretrained:
            if not local_checkpoint_path or not os.path.isfile(local_checkpoint_path):
                raise FileNotFoundError(
                    f"Local Whisper checkpoint not found at: {local_checkpoint_path}"
                )
            sd = torch.load(local_checkpoint_path, map_location="cpu")

            # unwrap common nesting
            if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                sd = sd["model"]
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]

            # Accept either full-model keys (encoder.*) or encoder-only keys
            enc_sd = {}
            for k, v in sd.items():
                if k.startswith("model.encoder."):          # common HF trainer prefix
                    enc_sd[k[len("model.encoder."):]] = v
                elif k.startswith("encoder."):              # full WhisperModel keys
                    enc_sd[k[len("encoder."):]] = v
                else:
                    # maybe already encoder-only keys (e.g., "layers.0.self_attn.q_proj.weight")
                    enc_sd[k] = v

            missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
            if strict_load and (missing or unexpected):
                raise RuntimeError(f"Strict load failed. missing={missing}, unexpected={unexpected}")
            print(f"[Whisper-Encoder] Loaded local weights: missing={len(missing)}, unexpected={len(unexpected)}")

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = self.encoder.config.d_model  # 512 for whisper-base
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )
        for p in self.head.parameters():
            p.requires_grad = True

        # Optional: force fixed channels (you were repeating to 64 later)
        self.to_fixed_channels = None  # e.g., nn.LazyConv2d(16, kernel_size=1)

    def _canon_input_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept:
          - (B, 1, T, F) -> (B, F, T)
          - (B, F, T)    -> (B, F, T)
          - (B, T) / (B,1,T) -> requires self.front_end
        Output: (B, 80, T')
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
            x = F.interpolate(x.unsqueeze(1), size=(80, T), mode='bilinear', align_corners=False).squeeze(1)

        return x  # (B,80,T)

    def forward(self, x: torch.Tensor):
        """
        x: your spectrogram (preferred) or raw (if front_end provided)
        Returns: (feats_4d, logits)
        """
        input_features = self._canon_input_features(x)       # (B, 80, T)

        # (Optional) clamp time length to fixed max for stability (kept from your code)
        T = input_features.shape[-1]
        if T < 3000:
            input_features = F.pad(input_features, (0, 3000 - T))
        elif T > 3000:
            input_features = input_features[..., :3000]

        out = self.encoder(input_features=input_features,
                           output_hidden_states=True,
                           return_dict=True)   # ensure we get hidden_states

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

        # Preserve your final shape tweak
        feats_4d = feats_4d.squeeze(3).unsqueeze(1)          # (B, 1, H, T)
        feats_4d = feats_4d.repeat(1, 64, 1, 1)               # (B, 64, H, T)

        return feats_4d, logits

