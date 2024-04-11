import torch.nn as nn

## Define a custom autoencoder model
class CustomAutoencoder(nn.Module):
    def __init__(self):
        super(CustomAutoencoder, self).__init__()
        self.normalize = nn.BatchNorm1d(784)
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 10),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 512),
            nn.Tanh(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.normalize(x).cuda()
        x = self.encoder(x)
        x = self.decoder(x)
        return x