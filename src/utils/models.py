import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.SiLU):
        super(Classifier, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_hidden(self, x, layer_idx):
        """
        Get the output of a specific hidden layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        layer_idx : int
            Index of the hidden layer to get output from.

        Returns
        -------
        torch.Tensor
            Output from the specified hidden layer.
        """
        # process input through the network up to the specified layer
        for i in range(layer_idx + 1):
            x = self.network[i](x)

        return x
