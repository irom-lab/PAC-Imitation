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
import scipy
import random
import matplotlib.pyplot as plt

from src.nn_grasp import PolicyNet
from src.grasp_rollout_env import GraspRolloutEnv
from src.pac_es import kl_inverse, compute_grad_ES


class TrainGrasp_bound:

	def __init__(self, L, 
					   obj_folder, 
					   result_path,
					   model_path,
					   load_train_path,
					   long_finger,
					   num_cpus):

		# Args
		self.L = L
		self.obj_folder = obj_folder
		self.result_path = result_path
		self.model_path = model_path
		self.load_train_path = load_train_path

		# Extract result
		info =	torch.load(result_path+'train_details')
		_, _, self.step_best_bound, self.mu_ps, self.logvar_ps, self.seed_states = info['best_bound_data']
		self.seed = self.seed_states[0]
		self.mu_pr = torch.zeros_like(self.mu_ps)
		self.logvar_pr = torch.zeros_like(self.logvar_ps)
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
		self.train_obj_ind_list = np.arange(0,500)
		self.test_obj_ind_list = np.arange(500,1000)

		# Load prior policy
		actor_pr = PolicyNet(input_num_chann=1,
							dim_mlp_append=0,
							num_mlp_output=5,
							out_cnn_dim=self.out_cnn_dim,
							z_conv_dim=self.z_conv_dim,
							z_mlp_dim=self.z_mlp_dim).to(device)
		actor_pr.load_state_dict(torch.load(self.actor_pr_path, map_location=device))
		for name, param in actor_pr.named_parameters():
			param.requires_grad = False
		actor_pr.eval()

		# Initialize rollout environment
		self.rollout_env = GraspRolloutEnv(
	  						actor=actor_pr, 
							z_total_dim=self.z_total_dim,
	   						num_cpus=num_cpus,
          					checkPalmContact=1,
              				useLongFinger=long_finger)


	def get_object_config(self, numTrials, obj_ind_list):
		obj_x = np.random.uniform(low=0.45, 
								  high=0.55, 
								  size=(numTrials, 1))
		obj_y = np.random.uniform(low=-0.05, 
								  high=0.05, 
								  size=(numTrials, 1))
		obj_yaw = np.random.uniform(low=-np.pi, high=np.pi, size=(numTrials, 1))
		objPos = np.hstack((obj_x, obj_y, 0.005*np.ones((numTrials, 1))))
		objOrn = np.hstack((np.zeros((numTrials, 2)), obj_yaw))
		objPathInd = np.arange(0,numTrials)  # each object has unique initial condition -> one env
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

		# Load finished samples if specified, train/test envs do not need to be loaded again
		if self.load_train_path is not "":
			checkpoint = torch.load(self.load_train_path)
			estimate_success_list = checkpoint['estimate_success_list']
			seed, python_seed_state, np_seed_state, torch_seed_state = checkpoint['seed_data']
			random.seed(seed)
			np.random.seed(seed)
			torch.manual_seed(seed)
			random.setstate(python_seed_state)
			np.random.set_state(np_seed_state)
			torch.set_rng_state(torch_seed_state)
		else:
			estimate_success_list = []
		sample_ind = int(len(estimate_success_list)/self.numTrainEnvs)

		# Run all samples
		while sample_ind <= self.L:
			sample_ind += 1
			print('\nRunning sample %d out of %d...\n' % (sample_ind, self.L))

			# Sample new latent every time
			epsilons = torch.normal(mean=0., std=1., 
							size=(self.numTrainEnvs, self.z_total_dim))
			sigma_ps = (0.5*logvar_ps).exp()
			zs_all = mu_ps + sigma_ps*epsilons

			success_list = self.rollout_env.parallel(
							zs_all=zs_all,
							objPos=objPos,
							objOrn=objOrn,
							objPathInd=objPathInd,
							objPathList=objPathList)
			estimate_success_list += success_list

			# Save every 100 samples
			if sample_ind % 100 == 0 and sample_ind > 0:
				torch.save({'estimate_success_list': estimate_success_list,
	   						'seed_data': (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()),
				   }, self.model_path+'estimate_train_'+str(sample_ind))

		estimate_cost = np.mean(array([1-s for s in estimate_success_list]))
		return estimate_cost


	def estimate_true_cost(self):
		# Extract envs
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
			estimate_success_list = self.rollout_env.parallel(
	   								zs_all=zs_all,
			   						objPos=objPos,
					 				objOrn=objOrn,
						 			objPathInd=objPathInd,
									objPathList=objPathList)
		estimate_cost =np.mean(array([1-s for s in estimate_success_list]))
		return estimate_cost


	def compute_final_bound(self):

		# Reload seed state at best bounds
		seed, python_seed_state, np_seed_state, torch_seed_state = self.seed_states
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		random.setstate(python_seed_state)
		np.random.set_state(np_seed_state)
		torch.set_rng_state(torch_seed_state)

		# Get estimated true cost using test envs
		true_estimate_cost = self.estimate_true_cost()

		# Get estimated train cost using trian envs and L=100
		train_estimate_start_time = time.time()
		train_estimate_cost = self.estimate_train_cost()
		train_estimate_time = time.time() - train_estimate_start_time

		# PAC-Bayes Reg
		mu_ps = self.mu_ps
		logvar_ps = self.logvar_ps
		mu_pr = self.mu_pr
		logvar_pr = self.logvar_pr

		# Get inverse bound
		_, R_final = self.get_pac_bayes(
	  						self.numTrainEnvs, 
							self.delta_final, 
							logvar_ps, 
				   			logvar_pr, 
					  		mu_ps, 
							mu_pr)
		cost_chernoff = kl_inverse(train_estimate_cost, 
							 	(1/self.L)*np.log(2/self.delta_prime))
		inv_bound = 1-kl_inverse(cost_chernoff, 2*R_final)

		# McAllester and Quadratic PAC Bound, use estimated training costs with L=100
		_, R = self.get_pac_bayes(
	  						self.numTrainEnvs,
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
		print("KL-inv bound:", inv_bound)
		print("Sample convergence cost:", cost_chernoff)
		print("Train estimate:", 1-train_estimate_cost)
		print("True estimate:", 1-true_estimate_cost)
		print('Time to run estimate training cost:', train_estimate_time)

		# Svae all the bounds
		np.savez(self.result_path+'L'+str(self.L)+'_step'+str(self.step_best_bound)+'_bounds.npz',
			R=R,
			maurer_bound=maurer_bound,
			quad_bound=quad_bound, 
			inv_bound=inv_bound,
			sample_convergence_cost=cost_chernoff,
			train_estimate=1-train_estimate_cost,
			true_estimate=1-true_estimate_cost,
			train_estimate_time=train_estimate_time,
			)


	def get_pac_bayes(self, N, delta, logvar_ps, logvar_pr, mu_ps, mu_pr):
		kld = (-0.5*torch.sum(1 \
							+ logvar_ps-logvar_pr \
							-(mu_ps-mu_pr)**2/torch.exp(logvar_pr) \
							-torch.exp(logvar_ps-logvar_pr))
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
	parser.add_argument('--AWS', type=int)
	parser.add_argument('--trial_name', type=str)
	parser.add_argument('--load_train_ind', type=int, default=0)
	parser.add_argument('--long_finger', type=int, default=1)
	arg_con = parser.parse_args()
	L = arg_con.L
	AWS = arg_con.AWS
	trial_name = arg_con.trial_name  # 'grasp_pac_1/'
	load_train_ind = arg_con.load_train_ind  # 'model/grasp_pac_1/estimate_train_100', 100 here
	long_finger = arg_con.long_finger
	obj_folder = '/home/ubuntu/SNC_v4_mug_xs/'
	num_cpus = 0  # no limit

	if load_train_ind == 0:  # not loading prev result
		load_train_path = ""
	else:
		load_train_path = 'model/'+trial_name+'estimate_train_'+str(load_train_ind)

	# Initialize trianing env
	with torch.no_grad():
		trainer = TrainGrasp_bound(
						L=L,  # to be varies
						obj_folder=obj_folder,  # different for AWS
						result_path='result/'+trial_name,
						model_path='model/'+trial_name,
						load_train_path=load_train_path,
						long_finger=long_finger,
	  					num_cpus=num_cpus)

		# Get bounds
		trainer.compute_final_bound()
