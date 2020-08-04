import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from numpy import array
import ray
import json
import time
import random

from src.nn_push import PolicyNet
from src.push_rollout_env import PushRolloutEnv
from src.pac_es import kl_inverse, compute_grad_ES


class TrainPush_bound:

	def __init__(self, L, 
					   obj_folder, 
					   result_path,
					   model_path,
					   num_cpus,
        			   x_range,
              		   y_range,
                   	   yaw_range,
                       target_y):

		# Args
		self.L = L
		self.obj_folder = obj_folder
		self.result_path = result_path
		self.model_path = model_path
		self.x_range = x_range
		self.y_range = y_range
		self.yaw_range = yaw_range

		# Extract result
		info = torch.load(result_path+'train_details')
		_, _, self.step_best_bound, self.mu_ps, self.logvar_ps, self.seed_state_best = info['best_bound_data']
		self.mu_pr = torch.zeros_like(self.mu_ps)
		self.logvar_pr = torch.zeros_like(self.logvar_ps)
		self.seed = self.seed_state_best[0]
		self.actor_pr_path = info['actor_pr_path']
		json_data = info['json_data']

		# Extract hyperparams from json_data
		config_dic, pac_dic, nn_dic, optim_dic = \
  			[value for key, value in json_data.items()]
		self.delta = pac_dic['delta']
		self.delta_prime = pac_dic['delta_prime']
		self.delta_final = pac_dic['delta_final']
		self.numTrainEnvs = pac_dic['numTrainEnvs']
		self.numTestEnvs = pac_dic['numTestEnvs']
		self.out_cnn_dim = nn_dic['out_cnn_dim']
		self.z_conv_dim = nn_dic['z_conv_dim']
		self.z_mlp_dim = nn_dic['z_mlp_dim']
		self.z_total_dim = nn_dic['z_conv_dim']+nn_dic['z_mlp_dim']

		# Get train/test envs from model
		checkpoint = torch.load(self.model_path+'model_5')
		self.trainEnvs = checkpoint['trainEnvs']
		self.testEnvs = checkpoint['testEnvs']		

		# Use CPU for ES for now
		device = 'cpu'

		# Config object index for all training and testing trials
		self.obj_folder = config_dic['obj_folder']
		self.train_obj_ind_list = np.arange(0,self.numTrainEnvs)
		self.test_obj_ind_list = np.arange(1000,1000+self.numTestEnvs)

		# Load prior policy, freeze params
		actor_pr = PolicyNet(input_num_chann=1,
							dim_mlp_append=10,  # eeHistory
							num_mlp_output=2,   # x/y action
							out_cnn_dim=self.out_cnn_dim,
							z_conv_dim=self.z_conv_dim,
							z_mlp_dim=self.z_mlp_dim,
	   						img_size=150).to(device)  # different from grasp
		actor_pr.load_state_dict(torch.load(self.actor_pr_path, map_location=device))
		for name, param in actor_pr.named_parameters():
			param.requires_grad = False
		actor_pr.eval()

		# Initialize rollout environment
		self.rollout_env = PushRolloutEnv(
	  						actor=actor_pr, 
							z_total_dim=self.z_total_dim,
			  				num_cpus=config_dic['num_cpus'],  # 0 for AWS
		 					y_target_range=target_y)


	def get_object_config(self, numTrials, obj_ind_list):
		obj_x = np.random.uniform(low=self.x_range[0], 
								  high=self.x_range[1], 
								  size=(numTrials, 1))
		obj_y = np.random.uniform(low=self.y_range[0], 
								high=self.y_range[1], 
								size=(numTrials, 1))
		obj_yaw = np.random.uniform(low=self.yaw_range[0], 
									high=self.yaw_range[1], 
									size=(numTrials, 1))
		objPos = np.hstack((obj_x, obj_y, 0.035*np.ones((numTrials, 1))))
		objOrn = np.hstack((np.zeros((numTrials, 2)), obj_yaw))
		objPathInd = np.arange(0,numTrials)
		objPathList = []
		for obj_ind in obj_ind_list:
			objPathList += [self.obj_folder + str(obj_ind) + '.urdf']
		return (objPos, objOrn, objPathInd, objPathList)


	def estimate_train_cost(self):
		# Extract envs
		objPos, objOrn, objPathInd, objPathList = self.trainEnvs

		# Posterior
		mu_ps = self.mu_ps.clone()
		logvar_ps = self.logvar_ps.clone()

		# Run all samples
		estimate_success_list = np.empty((0))
		for sample_ind in range(self.L):
			print('\nRunning sample %d out of %d...\n' % (sample_ind, self.L))

			# Sample new latent every time
			epsilons = torch.normal(mean=0., std=1., 
							size=(self.numTrainEnvs, self.z_total_dim))
			sigma_ps = (0.5*logvar_ps).exp()
			zs_all = mu_ps + sigma_ps*epsilons

			info = self.rollout_env.roll_parallel(
							zs_all=zs_all,
							objPos=objPos,
							objOrn=objOrn,
							objPathInd=objPathInd,
							objPathList=objPathList,
							mu=[0.3]*self.numTrainEnvs,
							sigma=[0.01]*self.numTrainEnvs,
							getTipPath=False)
			estimate_success_list = np.concatenate((estimate_success_list,
											array([s[0] for s in info])))

			# Save every 100 samples
			if sample_ind % 100 == 0 and sample_ind > 0:
				torch.save({'estimate_success_list': estimate_success_list,
	   						'seed_data': (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()),
				   }, self.model_path+'estimate_train_'+str(sample_ind))

		estimate_cost = np.mean(1-estimate_success_list)
		return estimate_cost


	def estimate_true_cost(self):
		# Extract envs
		self.numTestEnvs = 1000
		self.test_obj_ind_list = np.arange(1000,1000+self.numTestEnvs)
		self.testEnvs = self.get_object_config(numTrials=self.numTestEnvs, obj_ind_list=self.test_obj_ind_list)
		objPos, objOrn, objPathInd, objPathList = self.testEnvs

		# Posterior
		mu_ps = self.mu_ps.clone()
		logvar_ps = self.logvar_ps.clone()

		# Config all test trials
		epsilons = torch.normal(mean=0., std=1., 
						  	size=(self.numTestEnvs, self.z_total_dim))
		sigma_ps = (0.5*logvar_ps).exp()
		zs_all = mu_ps + sigma_ps*epsilons

		# Run test trials and get estimated true cost
		with torch.no_grad():  # speed up
			info = self.rollout_env.roll_parallel(
	   								zs_all=zs_all,
			   						objPos=objPos,
					 				objOrn=objOrn,
						 			objPathInd=objPathInd,
									objPathList=objPathList,
									mu=[0.3]*self.numTestEnvs,
									sigma=[0.01]*self.numTestEnvs,
									getTipPath=False)
			estimate_success_list = array([s[0] for s in info])
		estimate_cost = np.mean(1-estimate_success_list)
		return estimate_cost


	def compute_final_bound(self):

		# Reload seed state at best bounds
		seed, python_seed_state, np_seed_state, torch_seed_state = self.seed_state_best
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		random.setstate(python_seed_state)
		np.random.set_state(np_seed_state)
		torch.set_rng_state(torch_seed_state)

		# Get estimated true cost using test envs
		print('Estimating true cost...')
		true_estimate_cost = self.estimate_true_cost()
		print('True estimate:', true_estimate_cost)

		# Get estimated train cost using trian envs and L=100
		print('Estimating training cost (may take a while)...')
		train_estimate_start_time = time.time()
		train_estimate_cost = self.estimate_train_cost()
		print('\n\n\nTime to run estimate training cost:', time.time()-train_estimate_start_time)

		# PAC-Bayes Reg
		mu_ps = self.mu_ps
		logvar_ps = self.logvar_ps
		mu_pr = self.mu_pr
		logvar_pr = self.logvar_pr

		# Get inverse bound
		_, R_final = self.get_pac_bayes(self.numTrainEnvs, 
										self.delta_final, 
										logvar_ps, 
										logvar_pr, 
										mu_ps, 
										mu_pr)
		cost_chernoff = kl_inverse(train_estimate_cost, 
							 	(1/self.L)*np.log(2/self.delta_prime))
		final_bound = 1-kl_inverse(cost_chernoff, 2*R_final)

		# McAllester and Quadratic PAC Bound, use estimated training costs, L
		_, R = self.get_pac_bayes(self.numTrainEnvs,
								  self.delta,
								  logvar_ps,
								  logvar_pr,
								  mu_ps,
								  mu_pr)
		maurer_bound = 1-train_estimate_cost-np.sqrt(R)
		quad_bound = 1-(np.sqrt(train_estimate_cost + R) + np.sqrt(R))**2

		# Show results
		print('\n\n\nUsing params from step', self.step_best_bound)
		print('R:', R)
		print("Maurer Bound:", maurer_bound)
		print("Quadratic Bound:", quad_bound)
		print("Final bound:", final_bound)
		print("Train estimate:", 1-train_estimate_cost)
		print("True estimate:", 1-true_estimate_cost)

		# Svae all the bounds
		np.savez(self.result_path+'L'+str(self.L)+'_step'+str(self.step_best_bound)+'_bounds.npz',
			R=R,
			maurer_bound=maurer_bound,
			quad_bound=quad_bound, 
			final_bound=final_bound,
			train_estimate=1-train_estimate_cost,
			true_estimate=1-true_estimate_cost,
			)


	def get_pac_bayes(self, N, delta, logvar_ps, logvar_pr, mu_ps, mu_pr):
		kld = (-0.5*torch.sum(1 \
							+ logvar_ps-logvar_pr \
							-(mu_ps-mu_pr)**2/logvar_pr.exp() \
							-(logvar_ps-logvar_pr).exp())
		 		).item()  # as scalar
		R = (kld + np.log(2*np.sqrt(N)/delta))/(2*N)
		return kld, R  # as scalar, not tensor


if __name__ == '__main__':

	import argparse
	def collect_as(coll_type):
		class Collect_as(argparse.Action):
			def __call__(self, parser, namespace, values, options_string=None):
				setattr(namespace, self.dest, coll_type(values))
		return Collect_as
	parser = argparse.ArgumentParser(description='PAC-Bayes Opt')
	parser.add_argument('--L', type=int)
	parser.add_argument('--hard_task', type=int, default=1)
	parser.add_argument('--num_cpus', type=int, default=4)
	parser.add_argument('--trial_name', type=str)
	parser.add_argument('--obj_folder', type=str) # '/home/ubuntu/Box_v4/'
	arg_con = parser.parse_args()
	L = arg_con.L
	hard_task = arg_con.hard_task
	num_cpus = arg_con.num_cpus
	trial_name = arg_con.trial_name
	obj_folder = arg_con.obj_folder 

	# Configure task given difficulty
	if hard_task:
		x_range = [0.50, 0.65] 
		y_range = [-0.15, 0.15]
		yaw_range = [-np.pi/4, np.pi/4]
		target_y = 0.15
	else:
		x_range = [0.55, 0.65] 
		y_range = [-0.10, 0.10]
		yaw_range = [-np.pi/4, np.pi/4]
		target_y = 0.12

	# Initialize trianing env
	with torch.no_grad():
		trainer = TrainPush_bound(
						L=L,  # to be varies
						obj_folder=obj_folder,  # different for AWS
						result_path='result/'+trial_name,
						model_path='model/'+trial_name,
	  					num_cpus=num_cpus,
        				x_range=x_range,
            			y_range=y_range,
               			yaw_range=yaw_range,
                  		target_y=target_y)

		# Get bounds
		trainer.compute_final_bound()
