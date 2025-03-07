import torch
import torch.nn as nn


class FoldingNet(nn.Module):
    """
    "FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation"
    (https://arxiv.org/pdf/1712.07262.pdf)

    Attributes:
        num_pred: 16384
        latent_dim: 1024
        grid_size: 128
    """

    def __init__(self, latent_dim=1024, num_pred=16384):
        super().__init__()
        self.num_pred = num_pred
        self.latent_dim = latent_dim
        self.grid_size = int(pow(self.num_pred, 0.5) + 0.5)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.latent_dim + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        a = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2)  # (1, 2, num_pred)

    def forward(self, x):
        """
        Parameters
        ----------
            x: (B, N, 3), N = 2048
        
        Returns
        -------
            fd2: (B, num_pred, 3)
        """
        B, N, _ = x.shape
        
        # encoder
        feature = self.first_conv(x.transpose(2, 1))                             # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]              # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  512, N)
        feature = self.second_conv(feature)                                      # (B, 1024, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]             # (B, 1024)

        # decoder
        feature_global = feature_global.view(B, self.latent_dim, 1).expand(B, self.latent_dim, self.num_pred)
        seed = self.folding_seed.expand(B, 2, self.num_pred).to(x.device)

        fd1 = self.folding1(torch.cat([seed, feature_global], dim=1))
        fd2 = self.folding2(torch.cat([fd1, feature_global], dim=1))

        return fd2.transpose(2, 1).contiguous()


if __name__ == '__main__':
    model = FoldingNet()
    x = torch.randn(32, 2048, 3)
    y = model(x)
    print(y.shape)
