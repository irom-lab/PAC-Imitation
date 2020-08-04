import os
import sys
import warnings
warnings.filterwarnings("ignore")

import time
import json
import random
import numpy as np
from numpy import array
import torch
from torch import tensor
import visdom

from src.nn_func import reparameterize
from src.nn_push import Encoder_lstm, PolicyNet
from src.push_rollout_env import PushRolloutEnv
from dataset.datasetBCPush import train_dataset, test_dataset, my_collate


class TrainPush_BC:

	def __init__(self, json_file_name, result_path, model_path):
		# Store args
		self.json_file_name = json_file_name
		self.result_path = result_path
		self.model_path = model_path

		# Configure from JSON file
		with open(json_file_name+'.json') as json_file:
			self.json_data = json.load(json_file)
		config_dic, data_dic, nn_dic, loss_dic, optim_dic = [value for key, value in self.json_data.items()]
		self.N = config_dic['N']
		self.L = config_dic['L']
		self.B_split = config_dic['B_split']
		self.num_cpus = config_dic['num_cpus']
		self.numSeenTest = config_dic['numSeenTest']
		self.numUnseenTest = config_dic['numUnseenTest']
		self.test_y_target_range = config_dic['test_y_target_range']

		z_conv_dim = nn_dic['z_conv_dim']
		z_mlp_dim = nn_dic['z_mlp_dim']
		self.z_total_dim = z_conv_dim+z_mlp_dim
		state_dim = nn_dic['state_dim']
		action_dim = nn_dic['action_dim']
		lstm_hidden_dim = nn_dic['lstm_hidden_dim']
		self.include_extra_kl = nn_dic['include_extra_kl']  # left/middle/right
 
		# Set up seeding
		self.seed = 0
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)

		# Use GPU for BC
		device = 'cuda:0'
		self.device = device

		# Sample trials
		numTrainTrials = data_dic['numTrainTrials']
		numTestTrials = data_dic['numTestTrials']
		trainTrialsList = np.arange(0,numTrainTrials)
		testTrialsList = np.arange(numTrainTrials,numTrainTrials+numTestTrials)
		numTrain = len(trainTrialsList)-len(trainTrialsList)%self.N
		numTest = len(testTrialsList)
		print('Num of train trials: ', numTrain)
		print('Num of test trials: ', numTest)

		# Config object index for test_rollout trials
		self.obj_folder = config_dic['obj_folder']
		self.seen_obj_ind_list = np.arange(2000,2000+50)  # same as used in demo
		self.unseen_obj_ind_list = np.arange(2000-50,2000)

		# Create dataholder
		self.train_data = train_dataset(
	  								trainTrialsList, 
			  						config_dic['trainFolderDir'], 
									device='cpu')
		self.test_data = test_dataset(
									testTrialsList, 
		 							config_dic['testFolderDir'], 
									device='cpu')
		self.train_dataloader = torch.utils.data.DataLoader(
	  								self.train_data, 
			  						batch_size=self.N, 
									shuffle=True, 
									drop_last=True, 
						   			pin_memory=True, 
							  		num_workers=4, 
		   							collate_fn=my_collate,
				  					)
		self.test_dataloader = torch.utils.data.DataLoader(
	  								self.test_data,
									batch_size=self.N, 
		 							shuffle=False, 
									drop_last=True, 
					 				pin_memory=True, 
									num_workers=4, 
		   							collate_fn=my_collate,
				  					)  

		# Set up networks, calculate number of params
		self.encoder = Encoder_lstm(
								out_cnn_dim=nn_dic['encoder_out_cnn_dim'],
								dim_append=state_dim+action_dim,  # no action
								z_total_dim=self.z_total_dim,
								lstm_hidden_dim=lstm_hidden_dim,
								img_size=150).to(device)
		self.actor = PolicyNet(
	  							input_num_chann=1,
								dim_mlp_append=state_dim,
								num_mlp_output=action_dim,  # x/y only
								out_cnn_dim=nn_dic['actor_out_cnn_dim'],
								z_conv_dim=z_conv_dim,
								z_mlp_dim=z_mlp_dim,
								img_size=150).to(device)
		print('Num of actor parameters: %d' % sum(p.numel() for p in self.actor.parameters() if p.requires_grad))
		print('Num of encoder parameters: %d' % sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))

		# Set up optimizer
		self.optimizer = torch.optim.AdamW([
				{'params': self.actor.parameters(), 
	 			 'lr': optim_dic['actor_lr'], 
		 		 'weight_decay': optim_dic['actor_weight_decay']},
				{'params': self.encoder.parameters(), 
	 			 'lr': optim_dic['encoder_lr'], 
		 		 'weight_decay': optim_dic['encoder_weight_decay']}
				])
		if optim_dic['decayLR']['use']:
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
				optimizer, 
				milestones=optim_dic['decayLR']['milestones'], 
		  		gamma=optim_dic['decayLR']['gamma'])
		else:
			self.scheduler = None


	def forward_encoder(self, data_batch):
		# Extract data from batch
		(depth_batch, ee_batch, poseAction_batch) = data_batch
		depth_batch = depth_batch.to(self.device)  #Bx200x200, B for num_datapoints in a trial
		ee_batch = ee_batch.to(self.device).unsqueeze(0)  #NxBx3
		poseAction_batch = poseAction_batch.to(self.device).unsqueeze(0)  #NxBx3
		B = depth_batch.shape[0]

		# Forward pass, get single z for whole traj, LSTM
		mu, logvar = self.encoder(depth_batch,
						torch.cat((ee_batch, poseAction_batch), dim=2))  # 2 now since seq_len dim, output 1x(z_total_dim)
		return mu, logvar


	def forward_decoder(self, z_batch, B_all, data_batch, print_action=False):
		# Set up loss functions
		mse = torch.nn.MSELoss(reduction="sum")
		l1 = torch.nn.L1Loss(reduction="sum")

		# Extract data
		(mu, logvar) = z_batch # Nx(z_dim)
		(depth_batch, ee_batch, poseAction_batch) = data_batch 
		depth_batch = depth_batch.to(self.device)
		ee_batch = ee_batch.to(self.device)
		poseAction_batch = poseAction_batch.to(self.device)
		# (NxB_split)x150x150

		# Repeat data for L
		depth_batch = depth_batch.repeat_interleave(self.L, dim=0)
		ee_batch = ee_batch.repeat_interleave(self.L, dim=0)
		poseAction_batch = poseAction_batch.repeat_interleave(self.L, dim=0)
		mu = mu.repeat_interleave(self.L, dim=0)
		logvar = logvar.repeat_interleave(self.L, dim=0)
		B_all = B_all.repeat_interleave(self.L, dim=0)

		# Sample z for each traj, and repeat zs for all steps
		zs_train = reparameterize(mu, logvar)  # Nx(z_dim)
		zs_train = zs_train.repeat_interleave(B_all, dim=0)

		# Use same z as condition for action prediction at each timestep
		poseAction_pred = self.actor(img=depth_batch,
									zs=zs_train, 
									mlp_append=ee_batch)

		# Get losses
		trans_l2_loss = mse(poseAction_pred, poseAction_batch)
		trans_l1_loss = l1(poseAction_pred, poseAction_batch)
		return trans_l2_loss, trans_l1_loss


	def run(self, epoch, kl_weight, loss_dic, train=True):
		# To be divided by step size
		epoch_loss = 0
		epoch_trans_loss = 0
		epoch_kl_loss = 0

		# Switch NN mode
		if train:
			self.encoder.train()
			self.actor.train()
			data_loader = self.train_dataloader
		else:
			self.encoder.eval()
			self.actor.eval()
			data_loader = self.test_dataloader

		label_all = []
		mu_all = np.empty((0,self.z_total_dim))

		num_batch = 0
  
		for batch_ind, data_batch in enumerate(data_loader):
			'''
			In each batch (of trajectories), we backpropogate for num_bin_size times (max_seq_len//B_split). Each step has N*B_split*L datapoints (except for last one, residue).
			'''
			# Extract data
			depth_batch = data_batch[0]
			ee_batch = data_batch[1]
			poseAction_batch = data_batch[2]
			seq_len_batch = data_batch[3]
			label_batch = data_batch[4]  # list of scalars
			label_all += label_batch
			N = len(depth_batch)  # override
			T = sum(seq_len_batch)  # number of datapoints

			# split all datapoints randomly into bins with bin size = B_split
			trial_bins_all = []
			for trialInd in range(N):
				trial_bins = torch.split(torch.randperm(seq_len_batch[trialInd]), self.B_split)
				trial_bins_all += [trial_bins]
			num_steps = max([len(trial_bins) for trial_bins in trial_bins_all])

			# Initial losses of the batch
			batch_trans_l2_loss = tensor(0.0, requires_grad=True).to(self.device)
			batch_trans_l1_loss = tensor(0.0, requires_grad=True).to(self.device)

			# Get mu and logvar for each traj, has to use for loop as LSTM uses batch size 1, to avoid padding
			mu_total = torch.empty((0, self.z_total_dim), 
						requires_grad=True).to(self.device)
			logvar_total = torch.empty((0, self.z_total_dim), 
						requires_grad=True).to(self.device)
			for trialInd in range(N):  # each trial in the batch is a trajectory
				depth_trial = depth_batch[trialInd]  # Bx200x200
				ee_trial = ee_batch[trialInd]  # Bx3
				poseAction_trial = poseAction_batch[trialInd]  # Bx3

				# Get mu and logvar for the traj, 1x(z_dim)
				mu, logvar = self.forward_encoder(
					(depth_trial, ee_trial, poseAction_trial))

				# Add to batch, mu and logvar now Nx(z_dim)
				mu_total = torch.cat((mu_total, mu), dim=0)
				logvar_total = torch.cat((logvar_total, logvar), dim=0)
			mu_all = np.concatenate((mu_all, mu_total.clone().detach().cpu().numpy()))

			# Get KL losses for the batch, use mean
			batch_kl_loss = -0.5 * torch.mean(1 + logvar_total - mu_total.pow(2) - logvar_total.exp())

			# Get reconstruction loss by step
			for stepInd in range(num_steps):
				# Collect indices over N for the step as some traj may not have data in this step
				ind_all = [trial_bins_all[trialInd][stepInd] if len(trial_bins_all[trialInd]) > stepInd else torch.tensor([]) for trialInd in range(N)]
				step_B_all = tensor([len(s) for s in ind_all]).to(self.device)

				# Collect data for the step using indices for each traj
				depth_step = torch.cat([depth_batch[trialInd][ind] for trialInd, ind in enumerate(ind_all) if len(ind)>0], dim=0)
				ee_step = torch.cat([ee_batch[trialInd][ind] for trialInd, ind in enumerate(ind_all) if len(ind)>0], dim=0)
				poseAction_step = torch.cat([poseAction_batch[trialInd][ind] for trialInd, ind in enumerate(ind_all) if len(ind)>0], dim=0)

				# Run step thru decoder, where L samples taken for each, NxB_splitxL
				step_trans_l2_loss, step_trans_l1_loss = self.forward_decoder(
							(mu_total, logvar_total), 
							step_B_all, 
							(depth_step, ee_step, poseAction_step))

				# Add step losses to batch
				batch_trans_l2_loss += step_trans_l2_loss
				batch_trans_l1_loss += step_trans_l1_loss

			# Get training loss for the batch, averaged over each point in traj
			batch_trans_l2_loss /= self.L*T
			batch_trans_l1_loss /= self.L*T
			batch_train_loss = \
   				loss_dic['trans_l2_ratio']*batch_trans_l2_loss+\
							kl_weight*batch_kl_loss + \
							batch_trans_l1_loss

			# Backprop
			if train:
				# zero gradients, backward pass to get gradients
				self.optimizer.zero_grad()
				batch_train_loss.backward()

				# Clip gradient if specified
				if loss_dic['gradientClip']['use']:
					torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 
										loss_dic['gradientClip']['thres'])

				# Update weights using gradient
				self.optimizer.step()

			# Store loss
			epoch_loss += batch_train_loss.item()
			epoch_trans_loss += batch_trans_l1_loss.item()
			epoch_kl_loss += batch_kl_loss.item()
			num_batch += 1

		# Decay learning rate if specified
		if train and (self.scheduler is not None):
			self.scheduler.step()

		# Get batch average loss
		epoch_loss /= num_batch
		epoch_trans_loss /= num_batch
		epoch_kl_loss /= num_batch

		return epoch_loss, epoch_trans_loss, epoch_kl_loss


	def get_object_config(self, numTrials, obj_ind_list, high_y=False):
		x_range = [0.50, 0.65]
		y_range = [-0.15, 0.15]
		yaw_range = [-np.pi/4, np.pi/4]
		mu_range = [0.30, 0.30]
		sigma_range = [0.01, 0.01]
		obj_x = np.random.uniform(low=x_range[0], 
								  high=x_range[1], 
								  size=(numTrials, 1))
		if high_y:
			obj_y = np.concatenate((np.random.uniform(low=0.10, 
									high=0.15, 
									size=(int(numTrials/2), 1)),
									np.random.uniform(low=-0.15, 
									high=-0.10, 
									size=(int(numTrials/2), 1))))
		else:
			obj_y = np.random.uniform(low=y_range[0], 
									high=y_range[1], 
									size=(numTrials, 1))
		obj_yaw = np.random.uniform(low=yaw_range[0], 
									high=yaw_range[1], 
									size=(numTrials, 1))
		mu = np.random.uniform(low=mu_range[0], 
								  high=mu_range[1], 
								  size=(numTrials, 1))
		sigma = np.random.uniform(low=sigma_range[0], 
								high=sigma_range[1], 
								size=(numTrials, 1))
		objPos = np.hstack((obj_x, obj_y, 0.035*np.ones((numTrials, 1))))
		objOrn = np.hstack((np.zeros((numTrials, 2)), obj_yaw))
		objPathInd = np.random.randint(low=0,
								 	   high=len(obj_ind_list), 
									   size=numTrials)
		objPathList = []
		for obj_ind in obj_ind_list:
			objPathList += [self.obj_folder + str(obj_ind) + '.urdf']
		return (objPos, objOrn, objPathInd, objPathList, mu, sigma)


	def single_rollout(self, objPos=array([0.55, 0.07, 0.035]),
					  		objOrn=array([0., 0., 0.2]), 
							objInd=2010,
							mu=0.30,
							sigma=0.01,
							num_trials=1):
		# Initialize rollout env
		rollout_env = PushRolloutEnv(
	  						actor=self.actor.to('cpu'),
							z_total_dim=self.z_total_dim,
			  				num_cpus=self.num_cpus,
		 					y_target_range=self.test_y_target_range)

		# Run a trial with GUI
		zs_single = torch.normal(mean=0., std=1., size=(num_trials, self.z_total_dim))
		for ind in range(num_trials):
			info = rollout_env.roll_single(
					zs=zs_single[ind], 
					objPos=objPos, 
					objOrn=objOrn,
					objPath=self.obj_folder+str(objInd)+'.urdf',
					mu=mu,
					sigma=sigma)  # always use GUI
			success = info[0]  # since single trial

		# Move model back to GPU for training
		self.actor.to('cuda:0')

		return success


	def sample_zs(self, objPos, objOrn, objInd, mu, sigma, uniform=False):
		# Initialize rollout env
		rollout_env = PushRolloutEnv(
	  						actor=self.actor.to('cpu'),
							z_total_dim=self.z_total_dim,
			  				num_cpus=self.num_cpus,
		 					y_target_range=self.test_y_target_range)

		# Sample z
		if uniform:
			num_z = 21
			zs_all = np.arange(-1.0,1.1,0.1)[:, np.newaxis]
			zs_all = torch.from_numpy(np.repeat(zs_all, self.z_total_dim, 1)).float()  # same for all dims
		else:
			num_z = 5
			zs_all = torch.normal(mean=0.0, std=1.0, size=(num_z, self.z_total_dim))

		# Run all z
		info = rollout_env.roll_parallel(
						zs_all=zs_all,
						objPos=np.repeat(objPos.reshape(1,-1), num_z, 0),
						objOrn=np.repeat(objOrn.reshape(1,-1), num_z, 0),
						objPathInd=np.arange(num_z), 
						objPathList=[self.obj_folder+str(objInd)+'.urdf']*num_z,
						mu=[mu]*num_z,
						sigma=[sigma]*num_z,
	  					getTipPath=True)
		success_list = [s[0] for s in info]
		reward_list = [s[1] for s in info]
		path_list = [s[2] for s in info]

		# Plot all paths in GUI
		rollout_env.plot_trajs(objPos=objPos,
							   objOrn=objOrn,
							   objPath=self.obj_folder+str(objInd)+'.urdf',
							   success_list=success_list,
		  					   path_list=path_list)

		# Move model back to GPU for training
		self.actor.to('cuda:0')


	def test_rollout(self, epoch):
		# Initialize rollout env
		rollout_env = PushRolloutEnv(
	  						actor=self.actor.to('cpu'),
							z_total_dim=self.z_total_dim,
			  				num_cpus=self.num_cpus,
		 					y_target_range=self.test_y_target_range)

		# Get seen object configuration
		objPos, objOrn, objPathInd, objPathList, mu, sigma = \
	  						self.get_object_config(
	  							numTrials=self.numSeenTest, 
								obj_ind_list=self.seen_obj_ind_list)
		zs_all = torch.normal(mean=0, std=1, 
							size=(self.numSeenTest, self.z_total_dim))
		info = rollout_env.roll_parallel(
								zs_all=zs_all,
								objPos=objPos,
								objOrn=objOrn,
								objPathInd=objPathInd, 
								objPathList=objPathList,
								mu=mu,
			  					sigma=sigma,
				   				getTipPath=False)
		avg_success_seen = np.mean(array([s[0] for s in info]))

		# Get unseen object configuration
		objPos, objOrn, objPathInd, objPathList, mu, sigma = \
	  						self.get_object_config(
	  							numTrials=self.numUnseenTest, 
								obj_ind_list=self.unseen_obj_ind_list)
		zs_all = torch.normal(mean=0, std=1, 
						   	  size=(self.numUnseenTest, self.z_total_dim))
		info = rollout_env.roll_parallel(
								zs_all=zs_all,
								objPos=objPos,
								objOrn=objOrn,
								objPathInd=objPathInd, 
								objPathList=objPathList,
								mu=mu,
			  					sigma=sigma,
				   				getTipPath=False)
		avg_success_unseen = np.mean(array([s[0] for s in info]))

		# Get unseen (high y) object configuration
		objPos, objOrn, objPathInd, objPathList, mu, sigma = \
	  						self.get_object_config(
	  							numTrials=self.numUnseenTest, 
								obj_ind_list=self.unseen_obj_ind_list,
        						high_y=True)
		zs_all = torch.normal(mean=0, std=1, 
						   	  size=(self.numUnseenTest, self.z_total_dim))
		info = rollout_env.roll_parallel(
								zs_all=zs_all,
								objPos=objPos,
								objOrn=objOrn,
								objPathInd=objPathInd, 
								objPathList=objPathList,
								mu=mu,
			  					sigma=sigma,
				   				getTipPath=False)
		avg_success_unseen_high_y = np.mean(array([s[0] for s in info]))


		# Move model back to GPU for training
		self.actor.to('cuda:0')

		return avg_success_seen, avg_success_unseen, avg_success_unseen_high_y


	def load_model(self, model_path):
		# model_path = self.model_path+'bc_actor_100_old.pt'
		self.actor.load_state_dict(torch.load(model_path))


	def save_model(self, epoch):
		torch.save(self.actor.state_dict(), 
					self.model_path+'bc_actor_'+str(epoch)+'.pt')


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
	config_dic, data_dic, nn_dic, loss_dic, optim_dic = [value for key, value in json_data.items()]
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

	# Initialize trianing env
	trainer = TrainPush_BC(
	 				json_file_name=json_file_name, 
		 			result_path=result_path,
	  				model_path=model_path)

	if config_dic['visdom']:
		vis = visdom.Visdom(env='push_bc')
		trans_loss_window = vis.line(
			X=array([[0, 0]]),
			Y=array([[0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='Trans L1, '+json_file_name, 
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
						legend=['Seen', 'Unseen', 'Unseen high y']))

	# Training details to be recorded
	train_loss_list = []
	test_loss_list = []
	train_trans_loss_list = []
	test_trans_loss_list = []
	train_kl_loss_list = []
	test_kl_loss_list = []
	test_seen_success_list = []
	test_unseen_success_list = []
	test_unseen_high_y_success_list = []

	# Record best success rate on unseen model, to decide if save model
	best_unseen = 0

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
		epoch_loss, epoch_trans_loss, epoch_kl_loss = trainer.run(epoch=epoch, 
															loss_dic=loss_dic,
															kl_weight=kl_weight,
															train=True)
		train_loss_list += [epoch_loss]
		train_trans_loss_list += [epoch_trans_loss]
		train_kl_loss_list += [epoch_kl_loss]
		print('Epoch: %d, loss: %f, Trans: %.4f, KL: %.4f, KL weight: %.4f' % (epoch, epoch_loss, epoch_trans_loss, epoch_kl_loss, kl_weight))

		# Test
		with torch.no_grad():
			epoch_loss, epoch_trans_loss, epoch_kl_loss = trainer.run(
															epoch=epoch,
															loss_dic=loss_dic,
															kl_weight=kl_weight,
															train=False)
			test_loss_list += [epoch_loss]
			test_trans_loss_list += [epoch_trans_loss]
			test_kl_loss_list += [epoch_kl_loss]
			print('Test, loss: %f, trans: %.4f, KL: %.4f' % (epoch_loss, epoch_trans_loss, epoch_kl_loss))
			print('This epoch took: %.2f\n' % (time.time()-epoch_start_time))

			# Test success every ? epochs
			if (epoch % config_dic['test_freq'] == 0 or epoch == numEpochs-1) and epoch > 0:

				# Parallel
				sim_start_time = time.time()
				avg_success_seen, avg_success_unseen, avg_success_unseen_high_y = trainer.test_rollout(epoch=epoch)
				print('Time took to sim:', time.time() - sim_start_time)
				print('Seen/unseen success:', avg_success_seen, 
		  									  avg_success_unseen)
				test_seen_success_list += [avg_success_seen]
				test_unseen_success_list += [avg_success_unseen]
				test_unseen_high_y_success_list += [avg_success_unseen_high_y]

				# Save model
				if avg_success_unseen > (best_unseen-0.05):
					trainer.save_model(epoch)
					best_unseen = avg_success_unseen

				if config_dic['visdom']:
					vis.line(X=array([[epoch, epoch, epoch]]),
							Y=np.array([[test_seen_success_list[-1], 		
										test_unseen_success_list[-1],
          								test_unseen_high_y_success_list[-1]]]),
							win=success_window,update='append')

		# Visualize
		if config_dic['visdom']:
			vis.line(X=array([[epoch, epoch]]),
					 Y=array([[train_trans_loss_list[-1], 
							   test_trans_loss_list[-1]]]),
					win=trans_loss_window,update='append')
			vis.line(X=array([[epoch, epoch]]),
					Y=np.array([[train_kl_loss_list[-1], 
				  				 test_kl_loss_list[-1]]]),
					win=kl_loss_window,update='append')
