from torch import nn


class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: list[int]):
        super(DQN, self).__init__()

        self.q_net = self.create_network(input_size, output_size, hidden_layers)

    def create_network(
        self, input_size: int, output_size: int, hidden_layers: list[int]
    ):
        layers = []
        in_features = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            # layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())

            in_features = hidden_size

        # Output layer
        layers.append(nn.Linear(in_features, output_size))

        return nn.Sequential(*layers)

    def forward(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(0)

        return self.q_net(X)
