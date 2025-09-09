from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
import gymnasium as gym


class ResidualBlock(nn.Module):
    # Add dilation to the init method
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        # The padding must be equal to the dilation to keep the feature map size the same
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + shortcut)
        return x
    
class MyCnnPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space:gym.spaces.Dict ,features_dim=256):
        super().__init__(observation_space, features_dim)
        map_space = observation_space.spaces['map']
        n_input_channels = map_space.shape[0]

        self.cnn = nn.Sequential(
            ResidualBlock(n_input_channels, 32,dilation=1),
            ResidualBlock(32, 64,dilation=2),
            ResidualBlock(64, 128,dilation=4)
        )
        self.focus_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1), # From 128 channels down to 64
            nn.ReLU(),
        )


        self.semi_global_pool = nn.AdaptiveAvgPool2d((2, 2)) #We do that to get 4 global features (2x2 grid) up , down , left , right which are the movements possible

        with torch.no_grad():
            sample_input = torch.as_tensor(map_space.sample()[None]).float()
            cnn_output = self.cnn(sample_input) # Output has 128 channels

            # Get tactical features from the focused output (64 channels)
            focused_features = self.focus_conv(cnn_output)
            n_flattened_local_features = focused_features.flatten(1).shape[1]

            # Get strategic features from the original rich output (128 channels)
            pooled_output = self.semi_global_pool(cnn_output)
            n_quadrant_features = pooled_output.flatten(1).shape[1]



        steps_space_dim = observation_space.spaces['steps'].shape[0]

        self.steps_net = nn.Sequential(
            nn.Linear(steps_space_dim, 16),
            nn.ReLU(),
        )


        messages_space = observation_space.spaces['messages']
        max_agents = messages_space.shape[0]
        num_message_types = messages_space.high[0] + 1
        message_embedding_dim = 16

        self.message_embedding = nn.Embedding(num_message_types, message_embedding_dim)

        self.message_net = nn.Sequential(
            nn.Linear(max_agents * message_embedding_dim, 32),
            nn.ReLU()
        )

        combined_features_dim =  n_flattened_local_features + n_quadrant_features + 16 + 32 # Added 32 for messages and 16 for the message embedding dim

        self.linear = nn.Sequential(
            nn.Linear(combined_features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        cnn_features = self.cnn(observations['map'].float())

        # 2. Get Tactical (Local) features by flattening the entire map
        focused_local_features = self.focus_conv(cnn_features)
        local_features_flat = focused_local_features.flatten(1)

        # 3. Get Strategic (Quadrant) features by semi-global pooling
        pooled_features = self.semi_global_pool(cnn_features)
        quadrant_features_flat = pooled_features.flatten(1)

        # Process steps through its own small network
        steps_features = self.steps_net(observations['steps'])

        # Get embeddings for each message, result shape: (batch_size, max_agents, embedding_dim)
        """
        VecNormalize doesn't just chagge the data type; it also standardizes the data by subtracting the mean and dividing by the std.
        It is designed for continuous data 
        """
        message_indices = observations['messages'].long()
        message_indices = torch.clamp(message_indices, 0, self.message_embedding.num_embeddings - 1)
        message_embeddings = self.message_embedding(message_indices)

        # Flatten the embeddings to a single vector per batch item
        message_features_flat = message_embeddings.flatten(1)
        message_fetures = self.message_net(message_features_flat)


        # Concatenate the features from both sources
        combined_features = torch.cat((local_features_flat, quadrant_features_flat, steps_features, message_fetures), dim=1)
        return self.linear(combined_features)