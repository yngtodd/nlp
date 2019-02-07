import math

import torch
import torch.nn as nn
import gpytorch

from nlp.ml.models import CNN


class ConvFeatureExtractor(CNN):
    """Convolutional feature extractor."""
    def __init__(self):
        super(ConvFeatureExtractor, self).__init__()

    def forward(self, x):
         x = self.embedding(x).view(-1, 1, self.config.word_dim * self.config.max_sent_len)
         conv_results = []
         conv_results.append(self.features.conv1(x).view(-1, self.config.n_filters1))
         conv_results.append(self.features.conv2(x).view(-1, self.config.n_filters2))
         conv_results.append(self.features.conv3(x).view(-1, self.config.n_filters3))
         x = torch.cat(conv_results, 1)
         x = self.pool(x)
         return x


class GaussianProcessLayer(gpytorch.models.AdditiveGridInducingVariationalGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        super(GaussianProcessLayer, self).__init__(grid_size=grid_size, grid_bounds=[grid_bounds],
                                                   num_dim=num_dim, mixing_params=False, sum_output=False)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                log_lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1,
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        res = self.gp_layer(features)
        return res

