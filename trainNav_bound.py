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

from src.nn_nav import CNN_nav, Decoder_nav
from src.nav_rollout_env import NavRolloutEnv
from src.pac_es import kl_inverse, compute_grad_ES


class TrainNav_bound:

	def __init__(self, L, 
					   result_path,
					   model_path,
					   load_train_path):

		# Args
		self.L = L
		self.result_path = result_path
		self.model_path = model_path
		self.load_train_path = load_train_path

		# Extract result
		info = torch.load(result_path+'train_details')

		# Load data from best bound
		_, _, self.step_best_bound, self.mu_ps, self.logvar_ps, self.seed_state_best = info['best_bound_data']
		self.mu_pr = torch.zeros_like(self.mu_ps)
		self.logvar_pr = torch.zeros_like(self.logvar_ps)

		self.seed = self.seed_state_best[0]
		self.actor_pr_path = info['actor_pr_path']
		json_data = info['json_data']
		config_dic, pac_dic, nn_dic, optim_dic = \
  			[value for key, value in json_data.items()]

		self.delta = pac_dic['delta']
		self.delta_prime = pac_dic['delta_prime']
		self.delta_final = pac_dic['delta_final']
		self.numTrainEnvs = pac_dic['numTrainEnvs']
		self.numTestEnvs = pac_dic['numTestEnvs']
		self.include_reg = pac_dic['include_reg']

		dim_cnn_output = nn_dic['dim_cnn_output']
		dim_img_feat = 2*dim_cnn_output # combine RGB and depth
		self.z_dim = nn_dic['z_dim']

		self.actor_pr_path = config_dic['actor_pr_path']
		self.actor_pr_epoch = config_dic['actor_pr_epoch']
		self.saved_model_path = config_dic['saved_model_path']
		self.ES_method = config_dic['ES_method']
		self.collision_thres = config_dic['collision_thres']

		# Fixed
		self.config_version = 1
		self.max_rollout_steps = 120

		# Config for trials
		self.pose_range = [array([[-3.0, -1.0, -np.pi/2]]), 
					  	   array([[-1.0, 1.5, np.pi/2]])]
		self.train_table_id_range = np.arange(0,100)
		self.test_table_id_range = np.arange(100,200)
		self.train_chair_id_range = np.arange(0,90)
		self.test_chair_id_range = np.arange(90,180)
		self.table_folder = '/home/ubuntu/SNC_furniture/04379243_v2/'
		self.chair_folder = '/home/ubuntu/SNC_furniture/03001627_v2/'

		# Get train/test envs from model
		checkpoint = torch.load(self.model_path+'model_5')
		self.trainEnvs = checkpoint['trainEnvs']
		self.testEnvs = checkpoint['testEnvs']		

		# Use CPU for ES for now
		device = 'cpu'

		# Load prior policy, freeze params
		self.CNN = CNN_nav(dim_cnn_output=dim_cnn_output, 
					 		img_size=200).to(device)
		self.decoder = Decoder_nav(dim_img_feat=dim_img_feat,
									z_dim=self.z_dim,
									dim_output=4).to(device)
		CNN_load_path = self.actor_pr_path+'bc_CNN_'+ \
      						str(self.actor_pr_epoch)+'.pt'
		decoder_load_path = self.actor_pr_path+'bc_dec_'+ \
      						str(self.actor_pr_epoch)+'.pt'
		self.CNN.load_state_dict(torch.load(CNN_load_path, map_location=device))
		self.decoder.load_state_dict(torch.load(decoder_load_path, map_location=device))
		for name, param in self.CNN.named_parameters():
			param.requires_grad = False
		for name, param in self.decoder.named_parameters():
			param.requires_grad = False
		self.CNN.eval()  # not needed, but anyway
		self.decoder.eval()

		# Set again later
		self.rollout_env = NavRolloutEnv(CNN=self.CNN,
									decoder=self.decoder,
									z_dim=self.z_dim,
									max_rollout_steps=self.max_rollout_steps,
									num_cpus=96,
									num_gpus=8,
									batch_size=11,
									AWS=1,
									collision_thres=self.collision_thres,
									config_version=self.config_version)

	def estimate_train_cost(self):
		# Extract envs
		obj_poses_all, obj_paths_all = self.trainEnvs

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
			estimate_success_list = np.empty((0))
		sample_ind = int(len(estimate_success_list)/self.numTrainEnvs)
		print('Previous success:', np.mean(estimate_success_list))
		print('Num samples done:', sample_ind)

		# Run all samples
		while sample_ind <= self.L:
			sample_ind += 1

			print('\nRunning sample %d out of %d...\n' % (sample_ind, self.L))
			print('Avg:', 1-np.mean(estimate_success_list))

			# Sample new latent every time
			epsilons = torch.normal(mean=0., std=1., 
							size=(self.numTrainEnvs, self.z_dim))
			sigma_ps = (0.5*logvar_ps).exp()
			zs_all = mu_ps + sigma_ps*epsilons

			success = self.rollout_env.roll_parallel(zs_all=zs_all, 
												obj_poses_all=obj_poses_all,
												obj_paths_all=obj_paths_all)
			estimate_success_list = np.concatenate((estimate_success_list,
											array(success)))

			# Save every 100 samples
			if sample_ind % 100 == 0 and sample_ind > 0:
				torch.save({'estimate_success_list': estimate_success_list,
	   						'seed_data': (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()),
				   }, self.model_path+'estimate_train_'+str(sample_ind))

		estimate_cost = np.mean(1-estimate_success_list)
		return estimate_cost


	def estimate_true_cost(self):
		# Extract envs
		obj_poses_all, obj_paths_all = self.testEnvs

		# Posterior
		mu_ps = self.mu_ps.clone()
		logvar_ps = self.logvar_ps.clone()

		# Config all test trials
		epsilons = torch.normal(mean=0., std=1., 
						  	size=(self.numTestEnvs, self.z_dim))
		sigma_ps = (0.5*logvar_ps).exp()
		zs_all = mu_ps + sigma_ps*epsilons

		# Run test trials and get estimated true cost
		with torch.no_grad():  # speed up
			success = self.rollout_env.roll_parallel(zs_all=zs_all, 
												obj_poses_all=obj_poses_all,
												obj_paths_all=obj_paths_all)
		estimate_cost = np.mean(1-array(success))
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
		print('True estimate:', 1-true_estimate_cost)

		# Get estimated train cost using trian envs and L=100
		print('Estimating training cost (may take a while)...')
		train_estimate_start_time = time.time()
		train_estimate_cost = self.estimate_train_cost()
		print('\n\n\nTime to run estimate training cost:', time.time()-train_estimate_start_time)

		# Get inverse bound
		_, R_final = self.get_pac_bayes(self.numTrainEnvs, 
										self.delta_final, 
										logvar_ps, 
										logvar_pr, 
										mu_ps, 
										mu_pr)
		cost_chernoff = kl_inverse(train_estimate_cost, 
							 	(1/self.L)*np.log(2/self.delta_prime))
		inv_bound = 1-kl_inverse(cost_chernoff, 2*R_final)

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
		print("KL-inv bound:", inv_bound)
		print("Sample convergence cost:", cost_chernoff)
		print("Max KL-inv bound:", max_inv_bound)
		print("Train estimate:", 1-train_estimate_cost)
		print("True estimate:", 1-true_estimate_cost)

		# Svae all the bounds
		np.savez(self.result_path+'L'+str(self.L)+'_step'+str(self.step_best_bound)+'_bounds.npz',
			R=R,
			maurer_bound=maurer_bound,
			quad_bound=quad_bound, 
			inv_bound=inv_bound,
			sample_convergence_cost=cost_chernoff,
			max_inv_bound=max_inv_bound,
			train_estimate=1-train_estimate_cost,
			true_estimate=1-true_estimate_cost,
			)


	def get_pac_bayes(self, N, delta, logvar_ps, logvar_pr, mu_ps, mu_pr):
		kld = (-0.5*torch.sum(1 + logvar_ps-logvar_pr \
							-(mu_ps-mu_pr)**2/logvar_pr.exp() \
							-(logvar_ps-logvar_pr).exp())).item()  # as scalar
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
	parser.add_argument('--AWS', type=int, default=1)
	parser.add_argument('--trial_name', type=str)
	parser.add_argument('--load_train_ind', type=int, default=0)
	arg_con = parser.parse_args()
	L = arg_con.L
	AWS = arg_con.AWS
	trial_name = arg_con.trial_name  # 'grasp_pac_1/'
	load_train_ind = arg_con.load_train_ind  # 'model/grasp_pac_1/estimate_train_100', 100 here

	if load_train_ind == 0:  # not loading prev result
		load_train_path = ""
	else:
		load_train_path = 'model/'+trial_name+'estimate_train_'+str(load_train_ind)

	# Initialize trianing env
	with torch.no_grad():
		trainer = TrainNav_bound(
						L=L,  # to be varies
						result_path='result/'+trial_name,
						model_path='model/'+trial_name,
						load_train_path=load_train_path)

		# Get bounds
		trainer.compute_final_bound()
