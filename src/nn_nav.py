import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

from src.nn_func import SpatialSoftmax


class CNN_nav(nn.Module):
	def __init__(self,
				 dim_cnn_output=64,
				 img_size=200):
		super(CNN_nav, self).__init__()
		
		# CNN for both RGB and depth
		self.rgb_conv_1 = nn.Sequential(
								nn.Conv2d(in_channels=3,
				  						  out_channels=dim_cnn_output//8, 
				  						  kernel_size=5, stride=1, padding=2),
								# nn.BatchNorm2d(dim_cnn_output//4),
								nn.ReLU(),
								)

		self.rgb_conv_2 = nn.Sequential(
								nn.Conv2d(in_channels=dim_cnn_output//8, 
				  						  out_channels=dim_cnn_output//4, 
						  				  kernel_size=3, stride=1, padding=1),
								nn.ReLU(),
								)

		self.rgb_conv_3 = nn.Sequential(
								nn.Conv2d(in_channels=dim_cnn_output//4, 
				  						  out_channels=dim_cnn_output//2, 
						  				  kernel_size=3, stride=1, padding=1),
								nn.ReLU(),
								)

		self.rgb_conv_4 = nn.Sequential(
								nn.Conv2d(in_channels=dim_cnn_output//2, 
				  						  out_channels=dim_cnn_output//2, 
						  				  kernel_size=3, stride=1, padding=1),
								nn.ReLU(),
								)

		self.depth_conv_1 = nn.Sequential(
								nn.Conv2d(in_channels=1,
				  						  out_channels=dim_cnn_output//4, 
				  						  kernel_size=5, stride=1, padding=2),
								nn.ReLU(),
								)

		self.depth_conv_2 = nn.Sequential(
								nn.Conv2d(in_channels=dim_cnn_output//4, 
				  						  out_channels=dim_cnn_output//2, 
						  				  kernel_size=3, stride=1, padding=1),
								nn.ReLU(),
								)

		self.depth_conv_3 = nn.Sequential(
								nn.Conv2d(in_channels=dim_cnn_output//2, 
				  						  out_channels=dim_cnn_output//2, 
						  				  kernel_size=3, stride=1, padding=1),
								nn.ReLU(),
								)

		# Spatial softmax, output 64 (32 features x 2d pos)
		self.sm = SpatialSoftmax(height=img_size, 
						   		 width=img_size, 
							  	 channel=dim_cnn_output//2)


	def forward(self, img_seq):
		# img_sep: Nx3x4xHxW
		seq = False

		# Put sequences into a single batch
		if len(img_seq.shape) == 5:
			N, B, C, H, W = img_seq.shape
			img_batch = img_seq.reshape(N*B, C, H, W)
			seq = True
		else:
			N, C, H, W = img_seq.shape  # not sequence
			img_batch = img_seq
		rgb_batch = img_batch[:,:3,:,:]
		depth_batch = img_batch[:,3:,:,:]

		rgb_feat = self.rgb_conv_1(rgb_batch)
		rgb_feat = self.rgb_conv_2(rgb_feat)
		rgb_feat = self.rgb_conv_3(rgb_feat)
		rgb_feat = self.rgb_conv_4(rgb_feat)
		rgb_feat = self.sm(rgb_feat)
		if seq:
			rgb_feat = rgb_feat.reshape(N, B, -1)

		depth_feat = self.depth_conv_1(depth_batch)
		depth_feat = self.depth_conv_2(depth_feat)
		depth_feat = self.depth_conv_3(depth_feat)
		depth_feat = self.sm(depth_feat)
		if seq:
			depth_feat = depth_feat.reshape(N, B, -1)

		if seq:
			return torch.cat((rgb_feat, depth_feat), dim=2)
		else:
			return torch.cat((rgb_feat, depth_feat), dim=1)


class Encoder_nav(nn.Module):
	def __init__(self,
				 dim_img_feat=128,
				 dim_encoder_append=1,  # action
				 z_dim=12,
				 dim_lstm_hidden=20,
				):

		super(Encoder_nav, self).__init__()

		self.z_dim = z_dim

		# LSTM, input both img features, states, and actions, learn initial hidden states
		self.dim_lstm_hidden = dim_lstm_hidden
		self.num_lstm_layer = 1

		self.lstm = nn.LSTM(input_size=dim_img_feat+dim_encoder_append,
					  		hidden_size=self.dim_lstm_hidden,
							num_layers=self.num_lstm_layer,
						 	batch_first=True,
							bidirectional=False)  

		# LSTM and latent output
		self.linear_lstm = nn.Sequential(
									nn.Linear(self.dim_lstm_hidden,
				   							  self.dim_lstm_hidden,
											  bias=True),
									nn.ReLU(),
									)
		self.att_vec = Parameter(torch.randn((self.dim_lstm_hidden, 
								  		1), 
								 		dtype=torch.float).to('cuda:0'),requires_grad=True)
		self.m_enc = nn.Softmax(dim=1)  # along which dim
		self.enc_linear_out = nn.Linear(self.dim_lstm_hidden, 
										z_dim*2, # mu AND logvar
										bias=True) 


	def forward(self, img_feats_seq, action_feats_seq):

		# img_feats_seq: Nx3x(feat_dim)
		# action_seq: Nx3x1
		N, _, _ = img_feats_seq.shape

		# Combine img and action
		lstm_input = torch.cat((img_feats_seq, action_feats_seq), dim=2)

		# Pass concatenated feature timeseries into LSTM to get features at each step.
		hidden_a = torch.randn(self.num_lstm_layer, 
						 		N, # batch size
								self.dim_lstm_hidden).float().to('cuda:0')
		hidden_b = torch.randn(self.num_lstm_layer, 
						 		N, 
						   		self.dim_lstm_hidden).float().to('cuda:0')
		lstm_output, _ = self.lstm(lstm_input, 
							 	(hidden_a,hidden_b)) # NxBx(hidden_size) 

		# Attention
		lstm_output = self.linear_lstm(lstm_output.squeeze(0))  # -> NxBx(hidden_size)
		lstm_att = lstm_output@self.att_vec  # NxBx1
		lstm_att_softm = self.m_enc(lstm_att)  # NxBx1
		att_out = lstm_att_softm.permute(0,2,1)@lstm_output  # Nx1x(hidden_size)

		# Output
		out = self.enc_linear_out(att_out).squeeze(1)
		return out[:,:self.z_dim], out[:,self.z_dim:]


class Decoder_nav(nn.Module):
	def __init__(self,
				 dim_img_feat=128,
				 z_dim=12,
				 dim_output=3,
				):

		super(Decoder_nav, self).__init__()

		# Decoder MLP
		self.dec_linear_1 = nn.Sequential(
								nn.Linear(dim_img_feat+z_dim, 
				   						  dim_img_feat//2,
										  bias=True),
								nn.ReLU(),
								)
		self.dec_linear_2 = nn.Sequential(
									nn.Linear(dim_img_feat//2, 
											  dim_img_feat//2, 
										   	  bias=True),
									nn.ReLU(),
									)
		self.dec_linear_3 = nn.Linear(dim_img_feat//2, 
									 dim_output, 
									 bias=True)

		# Output softmax
		self.m_dec = nn.Softmax(dim=1)


	def forward(self, img_feats, zs):

		# Decoder MLP
		dec_input = torch.cat((img_feats, zs), dim=1)
		x = self.dec_linear_1(dec_input)
		x = self.dec_linear_2(x)
		x = self.dec_linear_3(x)
		out = self.m_dec(x)
  
		return out
