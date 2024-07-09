import torch
import torch.nn.functional as F
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_k, d_v):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, num_heads * d_k)
        self.W_K = nn.Linear(d_model, num_heads * d_k)
        self.W_V = nn.Linear(d_model, num_heads * d_v)

        self.W_O = nn.Linear(num_heads * d_v, d_model)

    def forward(self, Q, K, V):
        Q = self.W_Q(Q).view(Q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)  # B x h x L x d_k
        K = self.W_K(K).view(K.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)  # B x h x L x d_k
        V = self.W_V(V).view(V.size(0), -1, self.num_heads, self.d_v).transpose(1, 2)  # B x h x L x d_v

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # B x h x L x L
        attention_weights = F.softmax(scores, dim=-1)
        heads_output = torch.matmul(attention_weights, V)  # B x h x L x d_v

        concat_heads = heads_output.transpose(1, 2).contiguous().view(Q.size(0), -1, self.num_heads * self.d_v)
        multi_head_output = self.W_O(concat_heads)  # B x L x d_m
        return multi_head_output, attention_weights


class InputEncoder(nn.Module):
    def __init__(self, d_k, d_model):
        super(InputEncoder, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.encoding = nn.Linear(2 * d_k, d_model)

    def forward(self, x, matrix):
        concat_x = torch.cat([x, matrix], dim=2)
        return self.encoding(concat_x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_x, multi_head_output):
        return input_x + self.norm(multi_head_output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, d_k, d_v):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v)
        self.residual = ResidualConnection(d_model)
        self.feed_forward = FeedForwardBlock(d_model, d_ff)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        multi_head_output, attention_weights = self.self_attn(x, x, x)
        x = self.residual(x, multi_head_output)
        x = self.feed_forward(x)
        x = self.norm(x)
        return x, attention_weights


class CombineWeights(nn.Module):
    def __init__(self, seq_len, d_k):
        super(CombineWeights, self).__init__()
        self.seq_len = seq_len
        self.d_k = d_k
        self.linear = nn.Linear(seq_len, d_k)

    def forward(self, attention_weights):
        A = torch.mean(attention_weights, dim=1)
        y = torch.sigmoid(self.linear(A))
        return y


class Generator(nn.Module):
    def __init__(self, d_model, seq_len, d_ff, num_heads, d_k, d_v):
        super(Generator, self).__init__()
        self.input_encoding1 = InputEncoder(d_k, d_model)
        self.input_encoding2 = InputEncoder(d_k, d_model)

        self.positional_encoder1 = PositionalEncoding(d_model, seq_len)
        self.positional_encoder2 = PositionalEncoding(d_model, seq_len)

        self.transformer_block1 = TransformerBlock(d_model, d_ff, num_heads, d_k, d_v)
        self.transformer_block2 = TransformerBlock(d_model, d_ff, num_heads, d_k, d_v)

        self.linear1 = nn.Linear(d_model, d_k)
        self.linear2 = nn.Linear(d_model, d_k)
        self.combine_weights = CombineWeights(seq_len, d_k)

    def forward(self, x, time_lag, mask):
        out = self.input_encoding1(x, time_lag)
        out = self.positional_encoder1(out)
        out, _ = self.transformer_block1(out)
        x_tilda_1 = self.linear1(out)

        out = self.input_encoding2(x_tilda_1, mask)
        out = self.positional_encoder2(out)
        out, attention_weights = self.transformer_block2(out)
        x_tilda_2 = self.linear2(out)
        y = self.combine_weights(attention_weights)
        X_imputed = (1 - y) * x_tilda_1 + y * x_tilda_2
        return X_imputed


class Discriminator(nn.Module):
    def __init__(self, d_model, seq_len, d_ff, num_heads, d_k, d_v):
        super(Discriminator, self).__init__()
        self.input_encoding = InputEncoder(d_k, d_model)
        self.positional_encoder = PositionalEncoding(d_model, seq_len)
        self.transformer_block = TransformerBlock(d_model, d_ff, num_heads, d_k, d_v)
        self.linear = nn.Linear(d_model, d_k)

    def forward(self, x_imputed, hint):
        out = self.input_encoding(x_imputed, hint)
        out = self.positional_encoder(out)
        out, _ = self.transformer_block(out)
        out = self.linear(out)
        p_matrix = torch.sigmoid(out)
        return p_matrix


def calculate_matrices(x, mask_ratio=0.2):
    M_prime = ~torch.isnan(x).float()
    M = M_prime.clone()
    I = torch.zeros_like(M)

    # Artificially mask some observed values
    masked_indices = torch.rand_like(x) < mask_ratio
    M[masked_indices] = 0
    I[masked_indices] = 1
    return M, M_prime, I


def create_hint_matrix(M_prime, hint_ratio=0.9):
    hint_matrix = torch.zeros_like(M_prime)
    for i in range(M_prime.size(0)):
        observed_indices = (M_prime[i] == 1).nonzero().squeeze()
        hint_indices = torch.randperm(observed_indices.size(0))[:int(hint_ratio * observed_indices.size(0))]
        hint_matrix[i, observed_indices[hint_indices]] = 1
    return hint_matrix


def calculate_delta_matrix(x):
    # Calculate the time difference matrix delta
    delta = torch.zeros_like(x)
    for i in range(1, x.size(1)):
        delta[:, i, :] = 1 + delta[:, i - 1, :] * (1 - torch.isnan(x[:, i, :]).float())
    return delta


def impute_data(x, generator, discriminator, epochs=int(1e4), mask_ratio=0.2, hint_ratio=0.9, lr=0.001):
    generator_opt = torch.optim.Adam(generator.parameters(), lr=lr)
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=lr)
    M, M_prime, I = calculate_matrices(x, mask_ratio)
    delta = calculate_delta_matrix(x)
    for epoch in range(epochs):
        hint = create_hint_matrix(M_prime, hint_ratio)
        x_imputed = generator(x, delta, M)

        discriminator_opt.zero_grad()
        p_matrix = discriminator(x_imputed, hint)
        d_loss = F.mse_loss(p_matrix, M)
        d_loss.backward()
        discriminator_opt.step()

        generator_opt.zero_grad()
        x_imputed = generator(x, M, I)
        _ = discriminator(x_imputed, hint)

        term1 = F.mse_loss(x * M_prime, x_imputed * M_prime)
        term2 = F.mse_loss(x * I, x_imputed * I)

        g_loss = 0.5 * term1 + 0.5 * term2 - d_loss
        g_loss.backward()
        generator_opt.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}')
    return x_imputed
