import os
import sys
import warnings
warnings.filterwarnings("ignore")

import time
import json
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import torch
from torch import tensor
import visdom

from src.nn_func import reparameterize
from src.nn_nav import CNN_nav, Encoder_nav, Decoder_nav
from src.nav_rollout_env import NavRolloutEnv
from dataset.datasetBCNav import train_dataset, test_dataset


class TrainNav_BC:

	def __init__(self, json_file_name, result_path, model_path,      
		table_folder, chair_folder):

		# Store args
		self.json_file_name = json_file_name
		self.result_path = result_path
		self.model_path = model_path
		self.table_folder = table_folder
		self.chair_folder = chair_folder

		# Configure from JSON file
		with open(json_file_name+'.json') as json_file:
			self.json_data = json.load(json_file)
		config_dic, nn_dic, loss_dic, self.optim_dic = [value for key, value in self.json_data.items()]
		self.N = config_dic['N']
		self.L = config_dic['L']
		self.num_cpus = config_dic['num_cpus']
		self.numTest = config_dic['numTest']
		self.config_version = config_dic['config_version']
		self.max_rollout_steps = config_dic['max_rollout_steps']
		self.collision_thres = config_dic['collision_thres']

		self.z_dim = nn_dic['z_dim']
		dim_lstm_hidden = nn_dic['dim_lstm_hidden']
		dim_cnn_output = nn_dic['dim_cnn_output']
		dim_img_feat = 2*dim_cnn_output # combine RGB and depth

		# Set up seeding
		self.seed = 0
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)

		# Use GPU for BC
		device = 'cuda:0'
		self.device = device

		# Sample trials
		numTrainTrials = config_dic['numTrainTrials']
		numTestTrials = config_dic['numTestTrials']
		numTotalTrials = numTrainTrials+numTestTrials
		trainTrialsList = np.random.choice(range(numTotalTrials), numTrainTrials, replace=False)
		testTrialsList = list(set(range(numTotalTrials)) - set(trainTrialsList))
		numTrain = len(trainTrialsList)-len(trainTrialsList)%self.N
		numTest = len(testTrialsList)
		print('Num of train trials: ', numTrain)
		print('Num of test trials: ', numTest)

		# Create dataholder
		self.train_data = train_dataset(trainTrialsList, 
			  							config_dic['trainFolderDir'], 
										device='cpu')
		self.test_data = test_dataset(testTrialsList, 
		 								config_dic['testFolderDir'], 
										device='cpu')
		self.train_dataloader = torch.utils.data.DataLoader(
										self.train_data, 
										batch_size=self.N, 
										shuffle=True, 
										drop_last=True, 
										pin_memory=True, 
										num_workers=4, 
										)
		self.test_dataloader = torch.utils.data.DataLoader(
										self.test_data,
										batch_size=self.N, 
										shuffle=False, 
										drop_last=True, 
										pin_memory=True, 
										num_workers=4, 
										)  
				 
		# Set up networks, calculate number of params
		self.CNN = CNN_nav(dim_cnn_output=dim_cnn_output, 
					 	   img_size=200).to(device)
		self.encoder = Encoder_nav(dim_img_feat=dim_img_feat,
							 	   z_dim=self.z_dim,
								   dim_lstm_hidden=dim_lstm_hidden).to(device)
		self.decoder = Decoder_nav(dim_img_feat=dim_img_feat,
							 	   z_dim=self.z_dim,
								   dim_output=4).to(device)
		print('Num of CNN parameters: %d' % sum(p.numel() for p in self.CNN.parameters() if p.requires_grad))
		print('Num of encoder parameters: %d' % sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
		print('Num of decoder parameters: %d' % sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))

		if config_dic['resume_epoch'] > 0:
			self.load_model(config_dic['resume_epoch'], 
				   			config_dic['resume_path'])

		# Set up optimizer
		self.optimizer = torch.optim.AdamW([
				{'params': self.CNN.parameters(), 
	 			 'lr': self.optim_dic['cnn_dec_lr'], 
		 		 'weight_decay': self.optim_dic['weight_decay']},
				{'params': self.encoder.parameters(), 
	 			 'lr': self.optim_dic['enc_lr'], 
		 		 'weight_decay': self.optim_dic['weight_decay']},
				{'params': self.decoder.parameters(), 
	 			 'lr': self.optim_dic['cnn_dec_lr'], 
		 		 'weight_decay': self.optim_dic['weight_decay']},
				])
		if self.optim_dic['decayLR']['use']:
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
				optimizer, 
				milestones=self.optim_dic['decayLR']['milestones'], 
		  		gamma=self.optim_dic['decayLR']['gamma'])
		else:
			self.scheduler = None


	def run(self, epoch, loss_dic, kl_weight, train=True):

		# To be divided by batch size
		epoch_loss = 0
		epoch_prim_loss = 0
		epoch_kl_loss = 0
		num_batch = 0

		# Set up loss functions
		ce = torch.nn.CrossEntropyLoss(reduction="mean",
						weight=torch.tensor([1.,6.,3.,3.]).to('cuda:0'))

		# Switch NN mode
		if train:
			self.CNN.train()
			self.encoder.train()
			self.decoder.train()
			data_loader = self.train_dataloader
		else:
			self.CNN.eval()
			self.encoder.eval()
			self.decoder.eval()
			data_loader = self.test_dataloader

		num_batch = 0

		# Run all batches
		for batch_ind, data_batch in enumerate(data_loader):

			# Extract data
			rgbd_seq = data_batch[0].to(self.device)
			prim_seq = data_batch[1].to(self.device)
			N, B, C, H, W = rgbd_seq.shape  # 50x3x4x200x200
			prim_batch = prim_seq.reshape(N*B)  # long
			prim_seq = prim_seq.float().reshape(N,B,1)	# 50x3x1

			# Get image features in sequence
			img_feats_seq = self.CNN(rgbd_seq)  # NxBx(img_feat)

			# Get mu and logvar for whole sequences
			mu_total, logvar_total = self.encoder(img_feats_seq, prim_seq)

			# Get z and repeat for steps within sequences
			zs_train = reparameterize(mu_total, logvar_total)
			zs_train = zs_train.repeat_interleave(B, dim=0)

			# Get action for batch
			img_feats_batch = img_feats_seq.reshape(N*B, -1)
			prim_pred = self.decoder(img_feats_batch, zs_train)

			# Get losses
			batch_prim_loss = ce(prim_pred, prim_batch)
			batch_kl_loss = -0.5 * torch.mean(1 + logvar_total - mu_total.pow(2) - logvar_total.exp())
			batch_train_loss = kl_weight*batch_kl_loss + batch_prim_loss

			# Backprop
			if train:
				# zero gradients, perform a backward pass to get gradients
				self.optimizer.zero_grad()
				batch_train_loss.backward()

				# Clip gradient if specified
				if loss_dic['gradientClip']['use']:
					torch.nn.utils.clip_grad_norm_(self.CNN.parameters(), loss_dic['gradientClip']['thres'])
					torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), loss_dic['gradientClip']['thres'])
					torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), loss_dic['gradientClip']['thres'])

				# Update weights using gradient
				self.optimizer.step()

			# Store loss
			epoch_loss += batch_train_loss.item()
			epoch_prim_loss += batch_prim_loss.item()
			epoch_kl_loss += batch_kl_loss.item()
			num_batch += 1

		# Decay learning rate if specified
		if train and self.optim_dic['decayLR']['use']:
			self.scheduler.step()

		# Get batch average loss
		epoch_loss /= num_batch
		epoch_prim_loss /= num_batch
		epoch_kl_loss /= num_batch

		return epoch_loss, epoch_prim_loss, epoch_kl_loss


	def get_object_config(self, numTrials, train=True):

		# Choose numTrials from 1000
		chosen_ind = np.random.choice(a=np.arange(1000), size=numTrials, replace=False)

		# Read pre-generated envs
		if train:
			info = np.load('nav_bc_envs.npz')
		else:
			info = np.load('nav_train_envs.npz')  # use ES train as test data
		env_ind_all = info['env_ind_all']
		env_pose_all = info['env_pose_all']

		# Select envs based on chosen ind
		obj_poses_all = env_pose_all[chosen_ind]
		obj_paths_all = [(self.table_folder+str(env_ind_all[ind][0])+'/'+ \
									str(env_ind_all[ind][0])+'.obj',
						self.chair_folder+str(env_ind_all[ind][1])+'/'+ \
									str(env_ind_all[ind][1])+'.obj') \
                   		for ind in chosen_ind]
		return (obj_poses_all, obj_paths_all)


	def single_rollout(self, obj_poses, obj_paths, num_trials=1):

		# Initialize rollout env
		rollout_env = NavRolloutEnv(CNN=self.CNN.to('cpu'),
									decoder=self.decoder.to('cpu'),
									z_dim=self.z_dim,
									max_rollout_steps=self.max_rollout_steps,
         							collision_thres=self.collision_thres,
                					config_version=self.config_version)

		# Run a trial with GUI
		zs = torch.normal(mean=0., std=1., size=(num_trials, self.z_dim))
		for ind in range(num_trials):
			success = rollout_env.roll_single(zs=zs[ind].reshape(1,self.z_dim),
											  obj_poses=obj_poses,
											  obj_paths=obj_paths,
											  mode='pbgui')
			print('\nSuccess: %d\n' % success)

		zs = torch.normal(mean=0., std=1., size=(num_trials, 
										   		self.max_rollout_steps, 
												self.z_dim))
		for ind in range(num_trials):
			success = rollout_env.roll_single(zs=zs[ind:(ind+1),:,:],
											  obj_poses=obj_poses,
											  obj_paths=obj_paths,
											  mode='pbgui')
			print('\nSuccess: %d\n' % success)

		self.CNN.to('cuda:0')
		self.decoder.to('cuda:0')

		return success


	def sample_zs(self, obj_poses, obj_paths, num_z=5):
		# Initialize rollout env
		rollout_env = NavRolloutEnv(CNN=self.CNN.to('cpu'),
									decoder=self.decoder.to('cpu'),
									z_dim=self.z_dim,
			  						num_cpus=self.num_cpus,
									max_rollout_steps=self.max_rollout_steps,
                    				collision_thres=self.collision_thres,
                					config_version=self.config_version)

		# Same z for all steps
		zs_all = torch.normal(mean=0.0, std=1.0, size=(num_z, self.z_dim))
		rollout_env.plot_trajs(zs_all=zs_all,
								obj_poses_all=obj_poses,
								obj_paths_all=obj_paths)

		# Different z for steps
		zs_all = torch.normal(mean=0.0, std=1.0, size=(num_z, 
												 	  self.max_rollout_steps, 
													  self.z_dim))
		rollout_env.plot_trajs(zs_all=zs_all,
								obj_poses_all=obj_poses,
								obj_paths_all=obj_paths)

		self.CNN.to('cuda:0')
		self.decoder.to('cuda:0')


	def test_rollout(self, epoch, seen=True, random_z=False):
		# Initialize rollout env
		rollout_env = NavRolloutEnv(CNN=self.CNN.to('cpu'),
									decoder=self.decoder.to('cpu'),
									z_dim=self.z_dim,
			  						num_cpus=self.num_cpus,
									max_rollout_steps=self.max_rollout_steps,
									batch_size=10,
                    				collision_thres=self.collision_thres,
                					config_version=self.config_version)

		# Get seen object configuration
		if seen:
			obj_poses_all, obj_paths_all = self.get_object_config(
										numTrials=self.numTest, 
										train=True)
		else:
			obj_poses_all, obj_paths_all = self.get_object_config(
									numTrials=self.numTest, 
									train=False)

		# Configure z for all
		if random_z:
			zs_all=torch.normal(mean=0,std=1,size=(self.numTest,120,self.z_dim))
		else:
			zs_all = torch.normal(mean=0, std=1, size=(self.numTest,self.z_dim))

		# Run
		success_all = rollout_env.roll_parallel(zs_all=zs_all,
												obj_poses_all=obj_poses_all,
												obj_paths_all=obj_paths_all)
		self.CNN.to('cuda:0')
		self.decoder.to('cuda:0')
		return np.mean(success_all)


	def load_model(self, epoch, path):
		CNN_path = path+'bc_CNN_'+str(epoch)+'.pt'
		enc_path = path+'bc_enc_'+str(epoch)+'.pt'
		dec_path = path+'bc_dec_'+str(epoch)+'.pt'
		self.CNN.load_state_dict(torch.load(CNN_path))
		self.encoder.load_state_dict(torch.load(enc_path))
		self.decoder.load_state_dict(torch.load(dec_path))


	def save_model(self, epoch):
		torch.save(self.CNN.state_dict(), 
					self.model_path+'bc_CNN_'+str(epoch)+'.pt')
		torch.save(self.encoder.state_dict(), 
					self.model_path+'bc_enc_'+str(epoch)+'.pt')
		torch.save(self.decoder.state_dict(), 
					self.model_path+'bc_dec_'+str(epoch)+'.pt')


if __name__ == '__main__':

	# Fix seeds
	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# Read JSON config
	json_file_name = sys.argv[1]
	with open(json_file_name+'.json') as json_file:
		json_data = json.load(json_file)
	config_dic, nn_dic, loss_dic, optim_dic = [value for key, value in json_data.items()]
	numEpochs = config_dic['numEpochs']

	# Create a new subfolder under result
	result_path = 'result/'+json_file_name+'/'
	if not os.path.exists(result_path):
		os.mkdir(result_path)
		os.mkdir(result_path+'figure/')

	# Create a new subfolder under model
	model_path = 'model/'+json_file_name+'/'
	if not os.path.exists(model_path):
		os.mkdir(model_path)

	# Object path
	table_folder = '/home/ubuntu/data/processed_objects/SNC_furniture/04379243_v2/'
	chair_folder = '/home/ubuntu/data/processed_objects/SNC_furniture/03001627_v2/'

	# Initialize trianing env
	trainer = TrainNav_BC(json_file_name=json_file_name, 
							result_path=result_path,
							model_path=model_path,
	   						table_folder=table_folder,
			 				chair_folder=chair_folder)

	if config_dic['visdom']:
		vis = visdom.Visdom(env='nav_bc', port=8097)
		prim_loss_window = vis.line(
			X=array([[0, 0]]),
			Y=array([[0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='Prim, '+json_file_name, 
						legend=['Train Loss', 'Test Loss']))
		kl_loss_window = vis.line(
			X=array([[0, 0]]),
			Y=array([[0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='KL, '+json_file_name, 
						legend=['Train Loss', 'Test Loss']))
		success_window = vis.line(
			X=array([[0, 0, 0]]),
			Y=array([[0, 0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='Test success, '+json_file_name, 
						legend=['Seen', 'Unseen', 'Unseen random z']))

	# Training details to be recorded
	train_loss_list = []
	test_loss_list = []
	train_prim_loss_list = []
	test_prim_loss_list = []
	train_kl_loss_list = []
	test_kl_loss_list = []
	test_seen_success_list = []
	test_unseen_success_list = []
	test_unseen_random_z_success_list = []

	# Run
	for epoch in range(numEpochs):
	
		# Record time for each epoch
		epoch_start_time = time.time()
	
 		# KL annealing
		if epoch < loss_dic['kl_anneal_wait']:
			kl_weight = 0
		else:
			kl_weight = min((epoch-loss_dic['kl_anneal_wait']+1)/(loss_dic['kl_anneal_period']+1), 1.)*loss_dic['kl_loss_ratio']

		# Train
		epoch_loss, epoch_prim_loss, epoch_kl_loss = trainer.run(epoch=epoch, 
															loss_dic=loss_dic,
															kl_weight=kl_weight,
															train=True)
		train_loss_list += [epoch_loss]
		train_prim_loss_list += [epoch_prim_loss]
		train_kl_loss_list += [epoch_kl_loss]
		print('Epoch: %d, loss: %f, Prim: %.4f, KL: %.4f, KL weight: %.4f' % (epoch, epoch_loss, epoch_prim_loss, epoch_kl_loss, kl_weight))

		# Test
		with torch.no_grad():
			epoch_loss, epoch_prim_loss, epoch_kl_loss = trainer.run(
															epoch=epoch,
															loss_dic=loss_dic,
															kl_weight=kl_weight,
															train=False)
			test_loss_list += [epoch_loss]
			test_prim_loss_list += [epoch_prim_loss]
			test_kl_loss_list += [epoch_kl_loss]
			print('Test, loss: %f, Prim: %.4f, KL: %.4f' % (epoch_loss, epoch_prim_loss, epoch_kl_loss))
			print('This epoch took: %.2f\n' % (time.time()-epoch_start_time))

			# Test success every ? epochs
			if (epoch % config_dic['test_freq'] == 0 or epoch == numEpochs-1) and epoch > 0:

				# Clear GPU data for Gibson
				torch.cuda.empty_cache() 

				# Parallel
				sim_start_time = time.time()
				avg_success_seen = 0
				avg_success_unseen=trainer.test_rollout(epoch=epoch, seen=False)
				avg_success_unseen_random_z=trainer.test_rollout(epoch=epoch, seen=False, random_z=True)
				print('Time took to sim:', time.time() - sim_start_time)
				print('Seen/unseen success:', avg_success_seen, 
		  									  avg_success_unseen,
			   								  avg_success_unseen_random_z)
				test_seen_success_list += [avg_success_seen]
				test_unseen_success_list += [avg_success_unseen]
				test_unseen_random_z_success_list += [avg_success_unseen_random_z]

				# Save model for test freq
				trainer.save_model(epoch)

				if config_dic['visdom']:
					vis.line(X=array([[epoch, epoch, epoch]]),
							Y=np.array([[test_seen_success_list[-1], 		
									test_unseen_success_list[-1],
									test_unseen_random_z_success_list[-1]]]),
							win=success_window,update='append')

		# Add to Visdom
		if config_dic['visdom']:
			vis.line(X=array([[epoch, epoch]]),
					 Y=array([[train_prim_loss_list[-1], 
							   test_prim_loss_list[-1]]]),
					win=prim_loss_window,update='append')
			vis.line(X=array([[epoch, epoch]]),
					Y=np.array([[train_kl_loss_list[-1], 
								test_kl_loss_list[-1]]]),
					win=kl_loss_window,update='append')
