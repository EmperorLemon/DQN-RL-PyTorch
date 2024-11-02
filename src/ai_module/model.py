from torch import nn


class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: list[int]):
        super(DQN, self).__init__()
        
        # Board size is 4x4 
        
        # Possible tile values (0, 2, 4, 8, 
        #                       16, 32, 64, 128, 256, 512,
        #                       1024, 2048, 4096, 8192, 
        #                       16384, 32768)
        
        # Log2 (2^x) = (0, (0), 1 (2), 2 (4), 3 (8), 4 (16), 5 (32), 6 (64)
        #               7 (128), 8 (256), 9 (512), 
        #               10 (1024), 11 (2048), 12 (4096), 
        #               13 (8192), 14 (16384), 15 (32768))

        self.q_net = self.create_network(input_size, output_size, hidden_layers)

    def create_network(
        self, input_size: int, output_size: int, hidden_layers: list[int]
    ):
        layers = []
        in_features = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())

            in_features = hidden_size

        # Output layer
        layers.append(nn.Linear(in_features, output_size))

        return nn.Sequential(*layers)

    def forward(self, X):
        return self.q_net(X)
