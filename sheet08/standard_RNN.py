# Import packages
import pandas as pd
#from numpy.linalg import inv
##from scipy.optimize import root_scalar
from scipy.optimize import fsolve
import torch as tc
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

##%% MODEL DEFINITION
class LatentRNN(nn.Module):
    """
    Recurrent Neural Network for learning dynamical systems
    
    Architecture:
    z_t = tanh(C * x_{t-1} + W * z_{t-1} + h)  # Hidden state update
    \hat{x}_t = B * z_t + c                    # Output generation
    """
    
    def __init__(self, obs_dim, latent_dim, dropout=0.0):
        super(LatentRNN, self).__init__()

        self.obs_dim = obs_dim        # Dimension of observations (2 for sin/cos)
        self.latent_dim = latent_dim  # Dimension of hidden state z_t
   
        # 1. Input-to-hidden transformation: U matrix and bias b # Input-to-hidden (C * x_{t-1} + h)
        self.input_to_hidden = nn.Linear(obs_dim, latent_dim)
        # 2. Hidden-to-hidden transformation: V matrix  # Hidden-to-hidden (W * z_{t-1})
        self.hidden_to_hidden = nn.Linear(latent_dim, latent_dim, bias=False)
        # 3. Hidden-to-output transformation: W matrix and bias c # Hidden-to-output (B * z_t + c)
        self.hidden_to_output = nn.Linear(latent_dim, obs_dim)

        # Dropout 
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, time_series, h0):
        """
        Forward pass through the RNN
        
        Args:
            time_series: Input sequence of shape (seq_len, batch_size, obs_dim)
            h0: Initial hidden state of shape (1, batch_size, latent_dim)
            
        Returns:
            obs_output: Predicted observations of shape (seq_len, batch_size, obs_dim)
            h: Final hidden state of shape (1, batch_size, latent_dim)
        """

        seq_len, batch_size, obs_dim = time_series.size()
        h = h0.squeeze(0)
        
        outputs = []
        for t in range(seq_len):
            x_t = time_series[t]
            h = tc.tanh(self.input_to_hidden(x_t) + self.hidden_to_hidden(h))
            h = self.dropout(h)
            x_hat = self.hidden_to_output(h)
            outputs.append(x_hat)

        obs_output = tc.stack(outputs, dim=0)
        return obs_output, h.unsqueeze(0)

def train(model, data, learning_rate, moment=0, optimizer_function='SGD', print_loss=True, batch_size=1, batch_sequence_length=1, regul = None, epochs=1000):
    """
    Training function with configurable optimizers and mini-batching
    """
    
    if optimizer_function == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=moment)
    elif optimizer_function == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_function}")
    
    # Use Mean Squared Error (MSE) loss for this regression task
    loss_function = nn.MSELoss()
    
    losses = []
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Optimizer: {optimizer_function}, LR: {learning_rate}, Batch size: {batch_size}, Sequence length: {batch_sequence_length}")
    
    for epoch in range(epochs):
        # Create a tensor of shape (1, batch_size, hidden_size) with random values (the hidden states)
        h0 = tc.zeros((1, batch_size, model.latent_dim))
        
        # Prepare full sequences (input and target)
        x = data[:-1]  # Input: all timesteps except the last
        y = data[1:]   # Target: all timesteps except the first
        
        # Initialize tensors to hold batch data of shape (batch_sequence_length, batch_size, observation_dim):
        X = tc.empty((batch_sequence_length, batch_size, model.obs_dim))
        Y = tc.empty((batch_sequence_length, batch_size, model.obs_dim))
        
        for j in range(batch_size):
            # Sample a random starting index for the subsequence
            # Ensure: 0 <= ind <= len(x) - batch_sequence_length
            ind = tc.randint(0, len(x) - batch_sequence_length + 1, (1,)).item()
            
            # Extract subsequence and assign to batch tensors
            X[:, j, :] = x[ind: ind + batch_sequence_length]
            Y[:, j, :] = y[ind : ind + batch_sequence_length]
        
        # Forward pass
        # 1. Zero the gradients from previous iteration
        optimizer.zero_grad()
        # 2. Run the model forward pass with input X and initial hidden state h0
        output, _ = model.forward(X, h0)
        # 3. Calculate the loss between model output and target Y
        epoch_loss = loss_function(output, Y)
        if (regul != None): penalty = regul(model.parameters())
        else: penalty = 0
        epoch_loss = loss_function(output, Y) + penalty

        
        # Backward pass and optimization step
        # 1. Compute gradients via backpropagation
        epoch_loss.backward()
        # 2. Update model parameters
        optimizer.step()
        
        # Store loss for plotting
        losses.append(epoch_loss.item())
        
        # Print progress
        if epoch % 10 == 0 and print_loss:
            print(f"Epoch: {epoch} loss {epoch_loss.item():.6f}")
    
    return losses    