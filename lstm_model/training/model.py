# This is the main LSTM model and quantile head

import torch # for tensors and operators
import torch.nn as nn # for model components

class LSTMforecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        """ 
        input_size -> int: number of features per data point (for input layer)
        hidden_size -> int: number of neurons per layer (its controls the complexity)
        num_layers -> int: how many layers
        output_size -> int: For now it is 1, it will change later for the quantile part
        """
        super().__init__() # Adds the parent pytorch functionality into the file as default
        self.lstm() = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                              num_layers=num_layers, batch_first=True)
        """
        Batch_first = follows input method as (B,T,F)
        B = Batch size, T = timesteps, F = features per datapoint
        For training, it cuts down on computations
        During actual inference, only meant for parallelism
        B and T are dynamic and are inferenced in forward def. F is defined in our input size
        """

        self.fc = nn.Linear(hidden_size, output_size)
        # Mapping final hidden state = output_size

    def forward(self, x):
        """
        This function defines how the data flows through the layers. 
        It also will help us output our answer from the model.

        Input:
        x -> form of float array (FloatTensor): Holds (B,T,F) explained above

        Output:
        out -> Float: One number (for now) which is the contextual understanding of our output
        ie stock price
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Above line: lstm_out = grabs all the hidden states from all timesteps
        # h_n, c_n = c_n is the last current state and h_n is last hidden state
        # Both are 3d matrices, h_n = num_layers, batch size, neurons_per_layer
        last_hidden = h_n[-1]  # shape: (B, H) # 2d matrix with the input being applied to weight and bias later
        out = self.fc(last_hidden)  # shape: (B, 1) (Self.fc is the (neuron) it goes through (Wx+b))
        return out
    


        
        
        