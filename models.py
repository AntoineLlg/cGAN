import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGenerator(nn.Module):
    def __init__(self, n_classes, embedding_dim, latent_dim):
        super(ConditionalGenerator, self).__init__()
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        self.embed = nn.Embedding(n_classes, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + latent_dim, 2 * latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * latent_dim, 256 * 7 * 7),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )  # use dropout to prevent learning a single image per embedding

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
        )  # 256x14x14

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
        )  # 128x28x28

        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Tanh(),
        )  # 1x28x28

    def forward(self, x, labels):
        labels = self.embed(labels)
        x = torch.cat([x, labels], 1)
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_conv(x)
        return x


class Classificator(nn.Module):

    def __init__(self, out_channels, embedding_dim, n_classes):
        super(Classificator, self).__init__()
        self.embed = nn.Embedding(n_classes, embedding_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )  # 64

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )  # 512x7x7

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 32, 1),  # dimensionality reduction
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.emb_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 7 * 7 * 32),
        )

        # turn embedding into 7x7x32 tensor (learn spatial information)

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, labels):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        y = self.embed(labels)
        y = self.emb_layer(y)

        x = torch.cat([x, y.view(-1, 32, 7, 7)], 1)
        x = self.conv4(x)

        x = self.fc(x)

        return F.sigmoid(x)
