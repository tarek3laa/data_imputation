import numpy as np
from utils.gain_utils import *
import torch
import torch.nn as nn
import torch.optim as optim


class Gain:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def fit_transform(self, data_x, hint_rate=0.9, alpha=100, iterations=int(1e4)):
        # Define mask matrix
        data_m = 1 - np.isnan(data_x)

        # Other parameters
        no, dim = data_x.shape

        # Hidden state dimensions
        h_dim = int(dim)

        # Normalization
        norm_data, norm_parameters = normalization(data_x)
        norm_data_x = np.nan_to_num(norm_data, 0)

        class Generator(nn.Module):
            def __init__(self, dim, h_dim):
                super(Generator, self).__init__()
                self.fc1 = nn.Linear(dim * 2, h_dim)
                self.fc2 = nn.Linear(h_dim, h_dim)
                self.fc3 = nn.Linear(h_dim, dim)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x, m):
                inputs = torch.cat([x, m], dim=1)
                h1 = self.relu(self.fc1(inputs))
                h2 = self.relu(self.fc2(h1))
                g_prob = self.sigmoid(self.fc3(h2))
                return g_prob

        class Discriminator(nn.Module):
            def __init__(self, dim, h_dim):
                super(Discriminator, self).__init__()
                self.fc1 = nn.Linear(dim * 2, h_dim)
                self.fc2 = nn.Linear(h_dim, h_dim)
                self.fc3 = nn.Linear(h_dim, dim)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x, h):
                inputs = torch.cat([x, h], dim=1)
                h1 = self.relu(self.fc1(inputs))
                h2 = self.relu(self.fc2(h1))
                d_prob = self.sigmoid(self.fc3(h2))
                return d_prob

        generator = Generator(dim, h_dim)
        discriminator = Discriminator(dim, h_dim)

        G_solver = optim.Adam(generator.parameters())
        D_solver = optim.Adam(discriminator.parameters())

        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        # Start Iterations
        for it in range(iterations):
            # Sample batch
            batch_idx = sample_batch_index(no, self.batch_size)
            X_mb = norm_data_x[batch_idx, :]
            M_mb = data_m[batch_idx, :]
            # Sample random vectors
            Z_mb = uniform_sampler(0, 0.01, self.batch_size, dim)
            # Sample hint vectors
            H_mb_temp = binary_sampler(hint_rate, self.batch_size, dim)
            H_mb = M_mb * H_mb_temp

            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            X_mb = torch.tensor(X_mb, dtype=torch.float32)
            M_mb = torch.tensor(M_mb, dtype=torch.float32)
            H_mb = torch.tensor(H_mb, dtype=torch.float32)

            # Train Discriminator
            G_sample = generator(X_mb, M_mb)
            Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
            D_prob = discriminator(Hat_X, H_mb)

            D_loss = -torch.mean(M_mb * torch.log(D_prob + 1e-8) + (1 - M_mb) * torch.log(1. - D_prob + 1e-8))

            D_solver.zero_grad()
            D_loss.backward()
            D_solver.step()

            # Train Generator
            G_sample = generator(X_mb, M_mb)
            Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
            D_prob = discriminator(Hat_X, H_mb)

            G_loss = -torch.mean((1 - M_mb) * torch.log(D_prob + 1e-8))
            MSE_loss_val = mse_loss(M_mb * X_mb, M_mb * G_sample) / torch.mean(M_mb)

            G_loss_total = G_loss + alpha * MSE_loss_val

            G_solver.zero_grad()
            G_loss_total.backward()
            G_solver.step()

        ## Return imputed data
        Z_mb = uniform_sampler(0, 0.01, no, dim)
        M_mb = data_m
        X_mb = norm_data_x
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        X_mb = torch.tensor(X_mb, dtype=torch.float32)
        M_mb = torch.tensor(M_mb, dtype=torch.float32)

        imputed_data = generator(X_mb, M_mb).detach().numpy()
        imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

        # Renormalization
        imputed_data = renormalization(imputed_data, norm_parameters)

        return imputed_data
