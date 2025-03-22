import torch.nn as nn
import torch


class StyleInference(nn.Module):
    def __init__(self):
        super(StyleInference, self).__init__()

    def forward(self, _input):
        assert hasattr(self, "obs_normalizer")
        assert hasattr(self, "style_normalizer")

        assert hasattr(self, "actor")
        assert hasattr(self, "style_encoder")

        _obs, _style = _input

        _obs = self.obs_normalizer(_obs)
        _style = self.style_normalizer(_style)

        _latent_style = self.style_encoder(_style)
        _input_obs = torch.cat([_obs, _latent_style], dim=1)
        return self.actor(_input_obs)
