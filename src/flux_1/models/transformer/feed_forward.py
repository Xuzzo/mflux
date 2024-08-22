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
    def __init__(self, in_features, out_features, rank=4, network_alpha=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank)
        self.up = nn.Linear(rank, out_features)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
