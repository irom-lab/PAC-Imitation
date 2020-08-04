import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

from src.nn_func import SpatialSoftmax


class Encoder(nn.Module):
	def __init__(self,
				 input_num_chann=1,
				 dim_mlp_append=5,
				 out_cnn_dim=64,  # 32 features x 2 (x/y) = 64
				 z_total_dim=23,
				 img_size=128,
				):

		super(Encoder, self).__init__()

		self.z_total_dim = z_total_dim

		# CNN
		self.conv_1 = nn.Sequential(  # Nx1x128x128
								nn.Conv2d(in_channels=input_num_chann,
				  						  out_channels=out_cnn_dim//4, 
				  						  kernel_size=5, stride=1, padding=2),
								nn.ReLU(),
								)    # Nx16x128x128

		self.conv_2 = nn.Sequential(
								nn.Conv2d(in_channels=out_cnn_dim//4, 
				  						  out_channels=out_cnn_dim//2, 
						  				  kernel_size=3, stride=1, padding=1),
								nn.ReLU(),
								)    # Nx32x128x128

		# Spatial softmax, output 64 (32 features x 2d pos)
		self.sm = SpatialSoftmax(height=img_size, 
						   		 width=img_size, 
							  	 channel=out_cnn_dim//2)

		# MLP
		self.linear_1 = nn.Sequential(
									nn.Linear(out_cnn_dim+dim_mlp_append,
				   							  out_cnn_dim*2,
											  bias=True),
									nn.ReLU(),
									)

		self.linear_2 = nn.Sequential(
									nn.Linear(out_cnn_dim*2, 
											  out_cnn_dim*2, 
										   	  bias=True),
									nn.ReLU(),
									)

		# Output action
		self.linear_out = nn.Linear(out_cnn_dim*2, 
									z_total_dim*2, 
									bias=True) 


	def forward(self, img, mlp_append):
		if img.dim() == 3:
			img = img.unsqueeze(1)  # Nx1xHxW

		# CNN
		x = self.conv_1(img)
		x = self.conv_2(x)

		# Spatial softmax
		x = self.sm(x)

		# MLP
		x = self.linear_1(torch.cat((x, action), dim=1))
		x = self.linear_2(x)
		out = self.linear_out(x)

		return out[:,:self.z_total_dim], out[:,self.z_total_dim:] # mu, var



class PolicyNet(nn.Module):
	def __init__(self, 
				 input_num_chann=1, # not counting z_conv
				 dim_mlp_append=0, # not counting z_mlp
				 num_mlp_output=5,
				 out_cnn_dim=64,  # 32 features x 2 (x/y) = 64
				 z_conv_dim=4,
				 z_mlp_dim=4,
				 img_size=128,
				 ):

		super(PolicyNet, self).__init__()

		self.dim_mlp_append = dim_mlp_append
		self.num_mlp_output = num_mlp_output
		self.z_conv_dim = z_conv_dim
		self.z_mlp_dim = z_mlp_dim

		# CNN
		self.conv_1 = nn.Sequential(  # Nx1x128x128
								nn.Conv2d(in_channels=input_num_chann			+z_conv_dim,
				  						  out_channels=out_cnn_dim//4, 
				  						  kernel_size=5, stride=1, padding=2),
								nn.ReLU(),
								)    # Nx16x128x128

		self.conv_2 = nn.Sequential(
								nn.Conv2d(in_channels=out_cnn_dim//4, 
				  						  out_channels=out_cnn_dim//2, 
						  				  kernel_size=3, stride=1, padding=1),
								nn.ReLU(),
								)    # Nx32x128x128

		# Spatial softmax, output 64 (32 features x 2d pos)
		self.sm = SpatialSoftmax(height=img_size, 
                           		 width=img_size, 
                              	 channel=out_cnn_dim//2)

		# MLP
		self.linear_1 = nn.Sequential(
								nn.Linear(out_cnn_dim+dim_mlp_append+z_mlp_dim, 
				   						out_cnn_dim*2,
										bias=True),
								nn.ReLU(),
								)

		self.linear_2 = nn.Sequential(
									nn.Linear(out_cnn_dim*2, 
											  out_cnn_dim*2, 
										   	  bias=True),
									nn.ReLU(),
									)

		# Output action
		self.linear_out = nn.Linear(out_cnn_dim*2, 
									num_mlp_output, 
									bias=True) 


	def forward(self, img, zs, mlp_append=None):

		if img.dim() == 3:
			img = img.unsqueeze(1)  # Nx1xHxW
		N, _, H, W = img.shape

		# Attach latent to image
		if self.z_conv_dim > 0:
			zs_conv = zs[:,:self.z_conv_dim].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # repeat for all pixels, Nx(z_conv_dim)x200x200
			img = torch.cat((img, zs_conv), dim=1)  # along channel

		# CNN
		x = self.conv_1(img)
		x = self.conv_2(x)

		# Spatial softmax
		x = self.sm(x)

		# MLP, add latent as concat
		if self.z_mlp_dim > 0:
			x = torch.cat((x, zs[:,self.z_conv_dim:]), dim=1)
		if mlp_append is not None:
			x = torch.cat((x, mlp_append), dim=1)
   
		x = self.linear_1(x)
		x = self.linear_2(x)
		out = self.linear_out(x)

		return out

