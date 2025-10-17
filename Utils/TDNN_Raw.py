import torch
import torch.nn as nn

class TDNNRaw(nn.Module):
    """
    Raw-waveform TDNN backbone compatible with your existing wrapper:
      - has conv1..4, maxpool1..4, nonlinearity attributes
      - forward returns (feats_4d, logits)
    Input:  (B,T) or (B,1,T)   (mono)
    Output: feats_4d: (B, C, T', 1), logits: (B, num_class)
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_class: int = 4,
                 output_len: int = 1,
                 drop_p: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        
        self.conv1 = nn.Conv1d(1,  32, kernel_size=11, padding=5)
        self.bn1   = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3,  padding=1)
        self.bn2   = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3,  padding=1)
        self.bn3   = nn.BatchNorm1d(32)
        
        self.conv4 = nn.Conv1d(32,  4, kernel_size=3,  padding=1)
        self.bn4   = nn.BatchNorm1d(4)


        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.maxpool3 = nn.MaxPool1d(kernel_size=4, stride=2)
        self.maxpool4 = nn.MaxPool1d(kernel_size=8, stride=4)


        self.nonlinearity = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


        self.conv5   = nn.Conv1d(4, 256, kernel_size=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool1d(output_len)
        self.dropout = nn.Dropout(p=drop_p) if drop_p is not None else nn.Identity()
        self.fc      = nn.Linear(256 * output_len, num_class)

    def _canon_wave(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T) -> (B,1,T); (B,T,1)->(B,1,T); (B,C,T)->mixdown mono
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.shape[2] == 1 and x.shape[1] != 1:
                x = x.transpose(1, 2)
            if x.shape[1] != 1:
                x = x.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Expected (B,T) or (B,*,T); got {x.shape}")
        return x

    def forward(self, x):
        # robust to (B,T) or (B,1,T)
        x = self._canon_wave(x)     # -> (B,1,T)

        # Block 1
        x = self.conv1(x); x = self.bn1(x); x = self.nonlinearity(x)
        x = self.maxpool1(x)

        # Block 2
        x = self.conv2(x); x = self.bn2(x); x = self.nonlinearity(x)
        x = self.maxpool2(x)

        # Block 3
        x = self.conv3(x); x = self.bn3(x); x = self.nonlinearity(x)
        x = self.maxpool3(x)

        # Block 4
        x = self.conv4(x); x = self.bn4(x); x = self.nonlinearity(x)
        x = self.maxpool4(x)              # x: (B, 4, T')

        feats_4d = x.unsqueeze(-1)        # (B, 4, T', 1) for texture losses

        # Classification head
        h = self.conv5(x)                 # (B, 256, T')
        h = self.sigmoid(h)
        h = self.avgpool(h).flatten(1)    # (B, 256*output_len)
        h = self.dropout(h)
        logits = self.fc(h)               # (B, num_class)
        print(logits.shape)
        return logits
