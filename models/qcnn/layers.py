import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QuaternionConv

__all__ = ['QuaternionBatchNorm', 'QuaternionBatchNorm2d', 'QuaternionInstanceNorm2d', 'ShareSepConv', 'SmoothDilatedResidualBlock']

class QuaternionBatchNorm(torch.nn.Module):
    """This class implements the simplest form of a quaternion batchnorm as
    described in : "Quaternion Convolutional Neural Network for
    Color Image Classification and Forensics", Qilin Y. et al.
    Arguments
    ---------
    input_size : int
        Expected size of the dimension to be normalized.
    dim: int, optional
        Default: -1
        It defines the axis that should be normalized. It usually correspond to
        the channel dimension.
    gamma_init: float, optional
        Default: 1.0
        First value of gamma to be used (mean).
    beta_param: bool, optional
        Default: True
        When set to True the beta parameter of the BN is applied.
    momentum: float, optional
        Default: 0.1
        It defines the momentum as for the real-valued batch-normalization.
    eps: float, optional
        Default: 1e-4
        Term used to stabilize operation.
    track_running_stats: bool, optional
        Default: True
        Equivalent to the real-valued batchnormalization parameter.
        When True, stats are tracked. When False, solely statistics computed
        over the batch are used.
    Example
    -------
    >>> inp_tensor = torch.rand([10, 40])
    >>> QBN = QuaternionBatchNorm(input_size=40)
    >>> out_tensor = QBN(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40])
    """

    def __init__(
        self,
        input_size,
        dim=-1,
        gamma_init=1.0,
        beta_param=True,
        momentum=0.1,
        eps=1e-4,
        track_running_stats=True,
    ):
        super(QuaternionBatchNorm, self).__init__()

        self.num_features = input_size // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.momentum = momentum
        self.dim = dim
        self.eps = eps
        self.track_running_stats = track_running_stats

        self.gamma = Parameter(torch.full([self.num_features], self.gamma_init))
        self.beta = Parameter(
            torch.zeros(self.num_features * 4), requires_grad=self.beta_param
        )

        # instantiate moving statistics
        if track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(self.num_features * 4)
            )
            self.register_buffer("running_var", torch.ones(self.num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

    def forward(self, input):
        """Returns the normalized input tensor.
        Arguments
        ---------
        input : torch.Tensor (batch, time, [channels])
            input to normalize. It can be 2d, 3d, 4d.
        """

        exponential_average_factor = 0.0

        # Entering training mode
        if self.training:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1

            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = (
                    1.0 / self.num_batches_tracked.item()
                )
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

            # Get mean along batch axis
            mu = torch.mean(input, dim=0)
            mu_r, mu_i, mu_j, mu_k = torch.chunk(mu, 4, dim=self.dim)

            # Get variance along batch axis
            delta = input - mu
            delta_r, delta_i, delta_j, delta_k = torch.chunk(
                delta, 4, dim=self.dim
            )
            quat_variance = torch.mean(
                (delta_r ** 2 + delta_i ** 2 + delta_j ** 2 + delta_k ** 2),
                dim=0,
            )

            denominator = torch.sqrt(quat_variance + self.eps)

            # x - mu / sqrt(var + e)
            out = input / torch.cat(
                [denominator, denominator, denominator, denominator],
                dim=self.dim,
            )

            # Update the running stats
            if self.track_running_stats:
                self.running_mean = (
                    1 - exponential_average_factor
                ) * self.running_mean + exponential_average_factor * mu.view(
                    self.running_mean.size()
                )

                self.running_var = (
                    1 - exponential_average_factor
                ) * self.running_var + exponential_average_factor * quat_variance.view(
                    self.running_var.size()
                )
        else:
            q_var = torch.cat(
                [
                    self.running_var,
                    self.running_var,
                    self.running_var,
                    self.running_var,
                ],
                dim=self.dim,
            )
            out = (input - self.running_mean) / q_var

        # lambda * (x - mu / sqrt(var + e)) + beta

        q_gamma = torch.cat(
            [self.gamma, self.gamma, self.gamma, self.gamma], dim=self.dim
        )
        out = (q_gamma * out) + self.beta

        return out

class QuaternionBatchNorm2d(nn.Module):
    r"""Applies a 2D Quaternion Batch Normalization to the incoming data.
        """

    def __init__(self, num_features, gamma_init=1., beta_param=True, training=True):
        super(QuaternionBatchNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.training = training
        self.eps = torch.tensor(1e-5)

    def reset_parameters(self):
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        quat_components = torch.chunk(input, 4, dim=1)
        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
        delta_r, delta_i, delta_j, delta_k = r - torch.mean(r), i - torch.mean(i), j - torch.mean(j), k - torch.mean(k)
        quat_variance = torch.mean((delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2))
        denominator = torch.sqrt(quat_variance + self.eps)

        # Normalize
        r_normalized = delta_r / denominator
        i_normalized = delta_i / denominator
        j_normalized = delta_j / denominator
        k_normalized = delta_k / denominator

        beta_components = torch.chunk(self.beta, 4, dim=1)

        # Multiply gamma (stretch scale) and add beta (shift scale)
        new_r = (self.gamma * r_normalized) + beta_components[0]
        new_i = (self.gamma * i_normalized) + beta_components[1]
        new_j = (self.gamma * j_normalized) + beta_components[2]
        new_k = (self.gamma * k_normalized) + beta_components[3]

        new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)

        return new_input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma) \
               + ', beta=' + str(self.beta) \
               + ', eps=' + str(self.eps) + ')'

class QuaternionInstanceNorm2d(nn.Module):
    def __init__(self, num_features, gamma_init=1., beta_param=True, training=True):
        super(QuaternionInstanceNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.training = training
        self.eps = torch.tensor(1e-5)

    def reset_parameters(self):
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        quat_components = torch.chunk(input, 4, dim=1)
        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
        delta_r, delta_i, delta_j, delta_k = r - torch.mean(r, axis=[1, 2, 3], keepdim=True), i - torch.mean(i, axis=[1, 2, 3], keepdim=True), j - torch.mean(j, axis=[1, 2, 3], keepdim=True), k - torch.mean(k, axis=[1, 2, 3], keepdim=True)
        quat_variance = torch.mean((delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2))
        denominator = torch.sqrt(quat_variance + self.eps)

        # Normalize
        r_normalized = delta_r / denominator
        i_normalized = delta_i / denominator
        j_normalized = delta_j / denominator
        k_normalized = delta_k / denominator

        beta_components = torch.chunk(self.beta, 4, dim=1)

        # Multiply gamma (stretch scale) and add beta (shift scale)
        new_r = (self.gamma * r_normalized) + beta_components[0]
        new_i = (self.gamma * i_normalized) + beta_components[1]
        new_j = (self.gamma * j_normalized) + beta_components[2]
        new_k = (self.gamma * k_normalized) + beta_components[3]

        new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)

        return new_input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma) \
               + ', beta=' + str(self.beta) \
               + ', eps=' + str(self.eps) + ')'

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)

class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1, use_bn=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = get_norm(channel_num, use_bn)
        self.conv2 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = get_norm(channel_num, use_bn)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)

class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1, use_bn=False):
        super(SmoothDilatedResidualBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = get_norm(channel_num, use_bn=use_bn)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = QuaternionConv(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = get_norm(channel_num, use_bn=use_bn)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        return F.relu(x+y)


def get_norm(num_channels, use_bn=True):
    if use_bn:
        return QuaternionBatchNorm2d(num_features=num_channels)
    else:
        return QuaternionInstanceNorm2d(num_features=num_channels)

