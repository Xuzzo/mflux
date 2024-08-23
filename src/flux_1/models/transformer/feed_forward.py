from mlx import nn
import mlx.core as mx


class FeedForward(nn.Module):

    def __init__(self, activation_function):
        super().__init__()
        self.linear1 = nn.Linear(3072, 6144)
        self.linear2 = nn.Linear(6144, 3072)
        self.activation_function = activation_function

    def forward(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation_function(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.down = nn.Linear(in_features, 32, bias=False)
        self.up = nn.Linear(32, out_features, bias=False)

    def forward(self, hidden_states):
        down_hidden_states = self.down(hidden_states)
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states
