import os
import warnings
warnings.filterwarnings("ignore")

import torch
import time
import numpy as np
from numpy import array
import visdom
import json
import random
import matplotlib.pyplot as plt

from src.nn_grasp import Encoder, PolicyNet
from src.nn_func import reparameterize
from src.grasp_rollout_env import GraspRolloutEnv
from dataset.datasetBCGrasp import train_dataset, test_dataset


class TrainGrasp_BC:

	def __init__(self, json_file_name):

		# Configure from JSON file
		self.json_file_name = json_file_name
		with open(json_file_name+'.json') as json_file:
			self.json_data = json.load(json_file)
		config_dic, ent_dic, loss_dic, self.optim_dic = [value for key, value in self.json_data.items()]
		self.N = config_dic['N']
		self.num_cpus = config_dic['num_cpus']
		self.checkPalmContact = config_dic['checkPalmContact']
		self.useLongFinger = config_dic['useLongFinger']
		self.numTest = config_dic['numTest']
		numTrainTrials = config_dic['numTrainTrials']
		numTestTrials = config_dic['numTestTrials']
		z_conv_dim = ent_dic['z_conv_dim']
		z_mlp_dim = ent_dic['z_mlp_dim']
		self.z_total_dim = z_conv_dim+z_mlp_dim

		# Set up seeding
		self.seed = 0
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)

		# Use GPU for BC
		device = 'cuda:0'
		self.device = device

		# Sample trials
		trainTrialsList = np.arange(0,numTrainTrials)
		testTrialsList = np.arange(numTrainTrials,numTrainTrials+numTestTrials)
		numTrain = len(trainTrialsList)-len(trainTrialsList)%self.N
		numTest = len(testTrialsList)-len(testTrialsList)%self.N
		print('Num of train trials: ', numTrain)
		print('Num of test trials: ', numTest)

		# Config object index for success test trials
		self.obj_folder = config_dic['obj_folder']
		self.xy_range = 0.05
		self.seen_obj_ind_list = np.arange(1000,1000+60)
		self.unseen_obj_ind_list = np.arange(1000-20,1000)

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
							  		num_workers=4)
		self.test_dataloader = torch.utils.data.DataLoader(
	  								self.test_data,
									batch_size=self.N, 
		 							shuffle=False, 
									drop_last=True, 
					 				pin_memory=True, 
						 			num_workers=4)  # assume small test size, single batch

		# Set up networks, calculate number of params
		self.encoder = Encoder(out_cnn_dim=ent_dic['encoder_out_cnn_dim'],
								dim_mlp_append=config_dic['actionDim'],
								z_total_dim=self.z_total_dim,
								img_size=128).to(device)
		self.actor = PolicyNet(input_num_chann=1,
								dim_mlp_append=0,
								num_mlp_output=config_dic['actionDim'],
								out_cnn_dim=ent_dic['actor_out_cnn_dim'],
								z_conv_dim=z_conv_dim,
								z_mlp_dim=z_mlp_dim).to(device)
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


	def get_object_config(self, numTrials, obj_ind_list):
		obj_x = np.random.uniform(low=0.5-self.xy_range, 
								  high=0.5+self.xy_range, 
								  size=(numTrials, 1))
		obj_y = np.random.uniform(low=-self.xy_range, 
								  high=self.xy_range, 
								  size=(numTrials, 1))
		obj_yaw = np.random.uniform(low=-np.pi, high=np.pi, size=(numTrials, 1))
		objPos = np.hstack((obj_x, obj_y, 0.005*np.ones((numTrials, 1))))
		objOrn = np.hstack((np.zeros((numTrials, 2)), obj_yaw))
		objPathInd = np.random.randint(low=0, high=len(obj_ind_list), size=numTrials)  # use random ini cond for BC
		objPathList = []
		for obj_ind in obj_ind_list:
			objPathList += [self.obj_folder + str(obj_ind) + '.urdf']

		return (objPos, objOrn, objPathInd, objPathList)


	def forward(self, data_batch):

		# Set up loss functions
		mse = torch.nn.MSELoss(reduction="mean")
		l1 = torch.nn.L1Loss(reduction="mean")

		# Extract data from batch
		(depth_batch, eePos_batch, eeYaw_batch) = data_batch
		depth_batch = depth_batch.to(self.device)
		eePos_batch = eePos_batch.to(self.device)
		eeYaw_batch = eeYaw_batch.to(self.device).reshape(self.N,-1)

		# Forward pass
		mu, logvar = self.encoder(depth_batch, torch.cat((eePos_batch, eeYaw_batch), dim=1))
		zs_train = reparameterize(mu, logvar)
		pred = self.actor(depth_batch, zs_train)
		eePos_pred = pred[:,:3]
		eeYaw_pred = pred[:,3:5]  # supposed to be sin, cos

		# KL losses
		kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

		# Trans and Rot losses
		rot_l2_loss = mse(eeYaw_pred, eeYaw_batch)
		trans_l2_loss = mse(eePos_pred, eePos_batch)
		trans_l1_loss = l1(eePos_pred, eePos_batch)

		return kl_loss, rot_l2_loss, trans_l2_loss, trans_l1_loss


	def run(self, epoch, loss_dic, kl_weight, train=True):

		# To be divided by batch size
		epoch_loss = 0
		epoch_trans_loss = 0
		epoch_rot_loss = 0
		epoch_kl_loss = 0
		num_batch = 0

		# Switch NN mode
		if train:
			self.encoder.train()
			self.actor.train()
			data_loader = self.train_dataloader
		else:
			self.encoder.eval()
			self.actor.eval()
			data_loader = self.test_dataloader

		# Run all batches
		for _, data_batch in enumerate(data_loader):

			# Forward pass to get loss
			kl_loss, rot_l2_loss, trans_l2_loss, trans_l1_loss = self.forward(data_batch)

			# Get training loss
			train_loss = loss_dic['trans_l2_loss_ratio']*trans_l2_loss + \
							loss_dic['rot_l2_loss_ratio']*rot_l2_loss + \
							kl_weight*kl_loss + \
							trans_l1_loss

			if train:
				# zero gradients, perform a backward pass to get gradients
				self.optimizer.zero_grad()
				train_loss.backward()

				# Clip gradient if specified
				if loss_dic['gradientClip']['use']:
					torch.nn.utils.clip_grad_norm_(self.actor.parameters(), loss_dic['gradientClip']['thres'])

				# Update weights using gradient
				self.optimizer.step()

			# Store loss
			epoch_loss += train_loss.item()
			epoch_trans_loss += trans_l1_loss.item()
			epoch_rot_loss += rot_l2_loss.item()
			epoch_kl_loss += kl_loss.item()
			num_batch += 1

		# Decay learning rate if specified
		if train and self.optim_dic['decayLR']['use']:
			self.scheduler.step()

		# Get batch average loss
		epoch_loss /= num_batch
		epoch_trans_loss /= num_batch
		epoch_rot_loss /= num_batch
		epoch_kl_loss /= num_batch

		return epoch_loss, epoch_trans_loss, epoch_rot_loss, epoch_kl_loss


	def test_success(self, epoch, path):

		# Initialize rollout env
		rollout_env = GraspRolloutEnv(
	  						actor=self.actor.to('cpu'), 
							z_total_dim=self.z_total_dim,
			  				num_cpus=self.num_cpus,
		 					checkPalmContact=self.checkPalmContact,
        					useLongFinger=self.useLongFinger)

		# Run a trial with GUI, debug, save a latent interp figure
		zs_single = torch.normal(mean=0, std=1, 
						   		 size=(1, self.z_total_dim))
		success = rollout_env.single(
			zs=zs_single, 
			objPos=[0.5236,-0.03076,0.005], 
			objOrn=[0.,0.,-0.2988], 
			objPath=self.obj_folder+'1000.urdf', 
			gui=False, 
			save_figure=True, 
			figure_path=path+str(epoch)+'_z_interp')

		# Get seen object configuration
		objPos, objOrn, objPathInd, objPathList = self.get_object_config \
  								(numTrials=self.numTest, 
           						obj_ind_list=self.seen_obj_ind_list)
		zs_all = torch.normal(mean=0, std=1, 
						   	  size=(self.numTest, self.z_total_dim))
		success_list = rollout_env.parallel(
	  						zs_all=zs_all,
							objPos=objPos,
							objOrn=objOrn,
				   			objPathInd=objPathInd, 
					  		objPathList=objPathList)
		avg_success_seen = np.mean(array(success_list))

		# Get unseen object configuration
		objPos, objOrn, objPathInd, objPathList = self.get_object_config \
  								(numTrials=self.numTest, 
            					obj_ind_list=self.unseen_obj_ind_list)
		zs_all = torch.normal(mean=0, 
							  std=1, 
						   	  size=(self.numTest, self.z_total_dim))
		success_list = rollout_env.parallel(
	  						zs_all=zs_all,
							objPos=objPos,
							objOrn=objOrn,
				   			objPathInd=objPathInd, 
					  		objPathList=objPathList)
		avg_success_unseen = np.mean(array(success_list))

		# Move model back to GPU for training
		self.actor.to('cuda:0')

		return avg_success_seen, avg_success_unseen


	def save_model(self, epoch, path):
		torch.save(self.actor.state_dict(), 
					path+str(epoch)+'.pt')


if __name__ == '__main__':

	# Read JSON config
	json_file_name = sys.argv[1]
	with open(json_file_name+'.json') as json_file:
		json_data = json.load(json_file)
	config_dic, ent_dic, loss_dic, optim_dic = [value for key, value in json_data.items()]
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
	trainer = TrainGrasp_BC(json_file_name=json_file_name)

	if config_dic['visdom']:
		vis = visdom.Visdom(env='grasp')
		trans_loss_window = vis.line(
			X=array([[0, 0]]),
			Y=array([[0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='Trans L1, '+json_file_name, 
						legend=['Train Loss', 'Test Loss']))
		rot_loss_window = vis.line(
			X=array([[0, 0]]),
			Y=array([[0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='Rot L2, '+json_file_name, 
						legend=['Train Loss', 'Test Loss']))
		kl_loss_window = vis.line(
			X=array([[0, 0]]),
			Y=array([[0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='KL, '+json_file_name, 
						legend=['Train Loss', 'Test Loss']))
		accuracy_window = vis.line(
			X=array([[0, 0]]),
			Y=array([[0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='Test success rates, '+json_file_name, 
						legend=['Seen', 'Unseen']))

	# Training details to be recorded
	train_loss_list = []
	test_loss_list = []
	train_trans_loss_list = []
	test_trans_loss_list = []
	train_rot_loss_list = []
	test_rot_loss_list = []
	train_kl_loss_list = []
	test_kl_loss_list = []
	test_seen_accuracy_list = []
	test_unseen_accuracy_list = []

	# Record best success rate on unseen model, to decide if save model
	best_unseen = 0

	# Train
	for epoch in range(numEpochs):
	
		epoch_start_time = time.time()
	
		# KL annealing
		if epoch < loss_dic['kl_anneal_wait']:
			kl_weight = 0
		else:
			kl_weight = min((epoch-loss_dic['kl_anneal_wait'])/loss_dic['kl_anneal_period'], 1.)*loss_dic['kl_loss_ratio']

		# Run one pass of training
		epoch_loss, epoch_trans_loss, epoch_rot_loss, epoch_kl_loss = \
	  			trainer.run(epoch=epoch, 
							loss_dic=loss_dic, 
					  		kl_weight=kl_weight,
							train=True)
		train_loss_list += [epoch_loss]
		train_trans_loss_list += [epoch_trans_loss]
		train_rot_loss_list += [epoch_rot_loss]
		train_kl_loss_list += [epoch_kl_loss]
		print('Epoch: %d, loss: %f, Trans: %.4f, Rot: %.4f, KL: %.4f' % (epoch, epoch_loss, epoch_trans_loss, epoch_rot_loss, epoch_kl_loss))

		# Test sample trials
		with torch.no_grad():
			if epoch % 5 == 0 and epoch > 0:
				epoch_loss, epoch_trans_loss, epoch_rot_loss, epoch_kl_loss = \
							trainer.run(epoch=epoch, 
						 				loss_dic=loss_dic, 
										kl_weight=kl_weight,
							   			train=False)
				test_loss_list += [epoch_loss]
				test_trans_loss_list += [epoch_trans_loss]
				test_rot_loss_list += [epoch_rot_loss]
				test_kl_loss_list += [epoch_kl_loss]
				print('Test, loss: %f, trans: %.4f, rot: %.4f, KL: %.4f' % (epoch_loss, epoch_trans_loss, epoch_rot_loss, epoch_kl_loss))

		print('This epoch took: %.2f\n' % (time.time()-epoch_start_time))

		# Test success rate every 50 epochs
		with torch.no_grad():
			if (epoch % 50 == 0 or epoch == numEpochs-1) and epoch > 0:
				sim_start_time = time.time()
				avg_success_seen, avg_success_unseen = trainer.test_success(epoch=epoch,
					path=result_path+'figure/')

				print('Time took to sim:', time.time() - sim_start_time)
				print('Avg seen/unseen success rate:', avg_success_seen, avg_success_unseen)
				test_seen_accuracy_list += [avg_success_seen]
				test_unseen_accuracy_list += [avg_success_unseen]

				# Save model
				if avg_success_unseen > best_unseen-0.05:
					trainer.save_model(epoch, model_path)
					best_unseen = avg_success_unseen

				if config_dic['visdom']:
					vis.line(X=array([[epoch, epoch]]),
							Y=np.array([[test_seen_accuracy_list[-1], 		
				  				 		 test_unseen_accuracy_list[-1]]]),
							win=accuracy_window,update='append')

		# Visualize
		if config_dic['visdom']:
			vis.line(X=array([[epoch, epoch]]),
					 Y=array([[train_trans_loss_list[-1], 
							   test_trans_loss_list[-1]]]),
					win=trans_loss_window,update='append')
			vis.line(X=array([[epoch, epoch]]),
					Y=np.array([[train_rot_loss_list[-1], 
				  				 test_rot_loss_list[-1]]]),
					win=rot_loss_window,update='append')
			vis.line(X=array([[epoch, epoch]]),
					Y=np.array([[train_kl_loss_list[-1], 
				  				 test_kl_loss_list[-1]]]),
					win=kl_loss_window,update='append')
