import torch
import torch.nn as nn


class FC(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    The PCN encode and fully-connected decoder.

    Attributes:
        num_pred: 16384
    """
    def __init__(self, latent_dim=1024, num_pred=2048):
        super().__init__()

        self.num_pred = num_pred

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, latent_dim, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_pred * 3)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3)
        
        Returns
            x: (B, num_pred, 3)
        """
        B, N, _ = x.size()
        x = x.transpose(1, 2).contiguous()                        # (B, 3, N)

        # encode
        x = self.conv1(x)                                         # (B, 256, N)
        global_x = torch.max(x, dim=2, keepdim=True)[0]           #（B, 256, 1)
        x = torch.cat([global_x.expand(B, 256, N), x], dim=1)
        x = self.conv2(x)
        x = torch.max(x, dim=2)[0]

        # decode
        x = self.decoder(x).view(B, 3, self.num_pred)
        
        return x.transpose(1, 2).contiguous()
