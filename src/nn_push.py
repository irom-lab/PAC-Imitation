import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

from src.nn_func import SpatialSoftmax


class Encoder_lstm(nn.Module):
	def __init__(self,
				 input_num_chann=1,
				 dim_append=5,
				 out_cnn_dim=64,  # 32 features x 2 (x/y) = 64
				 z_total_dim=23,
				 lstm_hidden_dim=20,
				 img_size=128,
				):

		super(Encoder_lstm, self).__init__()

		self.z_total_dim = z_total_dim

		# CNN
		self.conv_1 = nn.Sequential(  # Nx1x128x128
								nn.Conv2d(in_channels=input_num_chann,
				  						  out_channels=out_cnn_dim//4, 
				  						  kernel_size=5, stride=1, padding=2),
                                # nn.BatchNorm2d(out_cnn_dim//4),
								nn.ReLU(),
								)    # Nx16x128x128

		self.conv_2 = nn.Sequential(
								nn.Conv2d(in_channels=out_cnn_dim//4, 
				  						  out_channels=out_cnn_dim//2, 
						  				  kernel_size=3, stride=1, padding=1),
                                # nn.BatchNorm2d(out_cnn_dim//2),
								nn.ReLU(),
								)    # Nx32x128x128

		# Spatial softmax, output 64 (32 features x 2d pos)
		self.sm = SpatialSoftmax(height=img_size, 
						   		 width=img_size, 
							  	 channel=out_cnn_dim//2)

		# LSTM, input both img features, states, and actions, learn initial hidden states
		self.lstm_hidden_dim = lstm_hidden_dim
		self.num_lstm_layer = 1

		self.lstm = nn.LSTM(input_size=out_cnn_dim+dim_append,
					  		hidden_size=self.lstm_hidden_dim,
							num_layers=self.num_lstm_layer,
						 	batch_first=True,
							bidirectional=False)  

		# MLP
		self.linear_lstm = nn.Sequential(
									nn.Linear(self.lstm_hidden_dim,
				   							  self.lstm_hidden_dim,
											  bias=True),
									nn.ReLU(),
									)

		# Attention for LSTM output
		self.att_vec = Parameter(torch.randn((self.lstm_hidden_dim, 
								  		1), 
								 		dtype=torch.float).to('cuda:0'),requires_grad=True)
		self.softm = nn.Softmax(dim=1)


		# Output
		self.linear_out = nn.Linear(self.lstm_hidden_dim, 
									z_total_dim*2, # mu AND logvar
									bias=True) 


	def forward(self, img_seq, mlp_append=None):

		if img_seq.dim() == 3:
			img_seq = img_seq.unsqueeze(1)  # Bx1xHxW
		B = img_seq.shape[0]  # seq_len

		# Pass each image thru CNN to get img spatial features timeseries
		x = self.conv_1(img_seq)
		x = self.conv_2(x)
		img_features = self.sm(x).unsqueeze(0)  # 1xBx32

		# Concatenate image features with state/actions before passing into LSTM
		if mlp_append is not None:
			lstm_input = torch.cat((img_features, mlp_append), dim=2) # add batch dimension, 1xBx(input_size)
		else:
			lstm_input = img_features
  
		# Pass concatenated feature timeseries into LSTM to get features at each step.
		hidden_a = torch.randn(self.num_lstm_layer, 
                         		1, # batch size
                        		self.lstm_hidden_dim).float().to('cuda:0')
		hidden_b = torch.randn(self.num_lstm_layer, 
                         		1, 
                           		self.lstm_hidden_dim).float().to('cuda:0')
		lstm_output, _ = self.lstm(lstm_input, 
                             	(hidden_a,hidden_b)) # 1xBx(hidden_size) 

		# Attention
		lstm_output = self.linear_lstm(lstm_output.squeeze(0))  # -> Bx(hidden_size)
		lstm_att = lstm_output@self.att_vec  # Bx1
		lstm_att_softm = self.softm(lstm_att)  # Bx1
		att_out = lstm_att_softm.T@lstm_output  # 1x(hidden_size)

		# Output
		out = self.linear_out(att_out)
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

		"""
		* Use spatial softmax instead of max pool by default
		"""
		
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

		self.linear_3 = nn.Sequential(
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
		x = self.linear_3(x)
		out = self.linear_out(x)

		return out
