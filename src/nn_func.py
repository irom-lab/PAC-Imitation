import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reparameterize(mu, logvar):
	std = torch.exp(0.5*logvar)
	eps = torch.randn_like(std)
	return mu + eps*std


class SpatialSoftmax(torch.nn.Module):
	def __init__(self, height, width, channel, data_format='NCHW'):
		super(SpatialSoftmax, self).__init__()
		self.data_format = data_format
		self.height = height
		self.width = width
		self.channel = channel

		pos_x, pos_y = np.meshgrid(
				np.linspace(-1., 1., self.height),
				np.linspace(-1., 1., self.width)
				)
		pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
		pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
		self.register_buffer('pos_x', pos_x)
		self.register_buffer('pos_y', pos_y)


	def forward(self, feature):
		# Output:
		#   (N, C*2) x_0 y_0 ...

		N = feature.shape[0]

		if self.data_format == 'NHWC':
			feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
		else:
			feature = feature.view(N, self.channel, self.height*self.width)

		softmax_attention = F.softmax(feature, dim=-1)

		# Sum over all pixels
		expected_x = torch.sum(self.pos_x*softmax_attention, dim=2, keepdim=False)
		expected_y = torch.sum(self.pos_y*softmax_attention, dim=2, keepdim=False)
		expected_xy = torch.cat([expected_x, expected_y], 1)

		return expected_xy
