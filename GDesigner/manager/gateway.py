# GDesigner/manager/gateway.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class VIBGateway(nn.Module):
    """
    Variational Information Bottleneck (VIB) Gateway.

    This module takes the raw, high-dimensional output from a fixed agent (API)
    and compresses it into a low-dimensional, task-relevant latent message 'm'.
    It also calculates the Information Bottleneck loss for regularization.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initializes the VIB Gateway.

        Args:
            input_dim (int): The dimension of the raw output embedding from an agent API.
            latent_dim (int): The dimension of the compressed latent message 'm'.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # The encoder network that maps raw input to the parameters of a Gaussian distribution
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, (input_dim + latent_dim) // 2),
            nn.ReLU(),
            nn.Linear((input_dim + latent_dim) // 2, latent_dim * 2)  # Output mu and log_var
        )

        print(f"[SKM-Net] VIBGateway initialized: InputDim={input_dim}, LatentDim={latent_dim}")

    def forward(self, raw_output_embedding: torch.Tensor):
        """
        Processes the raw output to generate a compressed message and the IB loss.

        Args:
            raw_output_embedding (torch.Tensor): The embedded raw output from an agent.
                                                  Shape: (batch_size, input_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - message (torch.Tensor): The sampled latent message 'm'. Shape: (batch_size, latent_dim)
            - ib_loss (torch.Tensor): The KL-divergence loss for this message. Shape: (scalar)
        """
        if raw_output_embedding.dim() == 1:
            raw_output_embedding = raw_output_embedding.unsqueeze(0)

        # Encode the raw output to get mu and log_var
        encoding = self.encoder(raw_output_embedding)
        mu, log_var = torch.chunk(encoding, 2, dim=-1)

        # Reparameterization trick to sample the message 'm'
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        message = mu + std * epsilon

        # --- Information Bottleneck Loss Calculation ---
        # The IB loss is the KL-divergence between the learned posterior distribution
        # q(m|raw_output) and a fixed prior p(m), which we set as a standard normal distribution N(0, I).
        #
        # Formula: L_IB = KL[q(m|raw_output) || p(m)]
        #
        # For two Gaussian distributions q = N(mu, sigma^2) and p = N(0, 1), the KL-divergence
        # has a convenient analytical form:
        # KL(q||p) = 0.5 * sum(log(1/sigma^2) + sigma^2 + mu^2 - 1)
        # This is equivalent to:
        # KL(q||p) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        #
        # This loss term encourages the learned message distribution 'q' to be as close
        # as possible to the simple, non-informative prior 'p', thus acting as a form of
        # regularization that penalizes informational complexity.
        ib_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

        # print(f"[SKM-Net Debug] VIBGateway: Input Shape={raw_output_embedding.shape}, "
        #       f"Message Shape={message.shape}, IB Loss={ib_loss.item():.4f}")

        return message, ib_loss