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


class TrainGrasp_PAC_ES:

	def __init__(self, json_file_name, result_path, model_path):

		# Extract JSON config
		self.json_file_name = json_file_name
		with open(json_file_name+'.json') as json_file:
			self.json_data = json.load(json_file)
		config_dic, pac_dic, nn_dic, optim_dic = \
  			[value for key, value in self.json_data.items()]

		self.delta = pac_dic['delta']
		self.delta_prime = pac_dic['delta_prime']
		self.delta_final = pac_dic['delta_final']
		self.numTrainEnvs = pac_dic['numTrainEnvs']
		self.numTestEnvs = pac_dic['numTestEnvs']
		self.L = pac_dic['L']
		self.include_reg = pac_dic['include_reg']

		self.out_cnn_dim = nn_dic['out_cnn_dim']
		self.z_conv_dim = nn_dic['z_conv_dim']
		self.z_mlp_dim = nn_dic['z_mlp_dim']
		self.z_total_dim = nn_dic['z_conv_dim']+nn_dic['z_mlp_dim']

		self.actor_pr_path = config_dic['actor_pr_path']
		self.numSteps = config_dic['numSteps']
		self.num_cpus = config_dic['num_cpus']
		self.saved_model_path = config_dic['saved_model_path']
		self.checkPalmContact = config_dic['checkPalmContact']
		self.ES_method = config_dic['ES_method']
		self.use_antithetic = config_dic['use_antithetic']
		self.num_epsilon = config_dic['num_epsilon']

		self.mu_lr = optim_dic['mu_lr']
		self.logvar_lr = optim_dic['logvar_lr']
		self.decayLR = optim_dic['decayLR']

		# Set up seeding
		self.seed = 0
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		# torch.backends.cudnn.deterministic = True
		# torch.backends.cudnn.benchmark = False

		# Use CPU for ES for now
		device = 'cpu'

		# Config object index for all training and testing trials
		self.obj_folder = config_dic['obj_folder']
		self.train_obj_ind_list = np.arange(0,self.numTrainEnvs)
		self.test_obj_ind_list = np.arange(500,500+self.numTestEnvs)

		# Load prior policy, freeze params
		actor_pr = PolicyNet(input_num_chann=1,
								dim_mlp_append=0,
								num_mlp_output=5,
								out_cnn_dim=self.out_cnn_dim,
								z_conv_dim=self.z_conv_dim,
								z_mlp_dim=self.z_mlp_dim).to(device)
		actor_pr.load_state_dict(torch.load(self.actor_pr_path, map_location=device))
		for name, param in actor_pr.named_parameters():
			param.requires_grad = False
		actor_pr.eval()  # not needed, but anyway

		# Initialize rollout environment
		self.rollout_env = GraspRolloutEnv(
	  						actor=actor_pr, 
							z_total_dim=self.z_total_dim,
			  				num_cpus=self.num_cpus,
         					checkPalmContact=self.checkPalmContact,
              				useLongFinger=config_dic['use_long_finger'])

		# Set prior distribution of parameters
		self.mu_pr = torch.zeros((self.z_total_dim))
		self.logvar_pr = torch.zeros((self.z_total_dim))

		# Initialize the posterior distribution
		self.mu_param = torch.tensor(self.mu_pr, requires_grad=True)
		self.logvar_param = torch.tensor(self.logvar_pr, requires_grad=True)

		# Get training envs
		self.trainEnvs = self.get_object_config(numTrials=self.numTrainEnvs, obj_ind_list=self.train_obj_ind_list)

		# Get test envs
		self.testEnvs = self.get_object_config(numTrials=self.numTestEnvs, obj_ind_list=self.test_obj_ind_list)

		# Recording: training details and results
		self.result_path = result_path
		self.model_path = model_path
		self.best_bound_data = (0, 0, 0, None, None, (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()))  # emp, bound, step, mu, logvar, seed
		self.best_emp_data = (0, 0, 0, None, None, (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state())) 
		self.cost_env_his = []  # history for plotting
		self.reg_his = []
		self.kl_his = []
		self.lr_his = []


	def get_object_config(self, numTrials, obj_ind_list):
		obj_x = np.random.uniform(low=0.45, 
								  high=0.55, 
								  size=(numTrials, 1))
		obj_y = np.random.uniform(low=-0.05, 
								  high=0.05, 
								  size=(numTrials, 1))
		obj_yaw = np.random.uniform(low=-np.pi,
                              		high=np.pi,
                                	size=(numTrials, 1))
		objPos = np.hstack((obj_x, obj_y, 0.005*np.ones((numTrials, 1))))
		objOrn = np.hstack((np.zeros((numTrials, 2)), obj_yaw))
		objPathInd = np.arange(0,numTrials)  # each object has unique initial condition -> one env
		objPathList = []
		for obj_ind in obj_ind_list:
			objPathList += [self.obj_folder + str(obj_ind) + '.urdf']
		return (objPos, objOrn, objPathInd, objPathList)


	def train(self):

		# Resume saved model if specified
		if self.saved_model_path is not "":
			checkpoint = torch.load(self.saved_model_path)
			start_step = checkpoint['step']

			# Retrieve
			self.best_bound_data = checkpoint['best_bound_data']
			self.best_emp_data = checkpoint['best_emp_data']
			self.cost_env_his = checkpoint['cost_env_his']
			self.reg_his = checkpoint['reg_his']
			self.kl_his = checkpoint['kl_his']
			self.lr_his = checkpoint['lr_his']

			# Update params
			self.mu_param = checkpoint['mu']
			self.logvar_param = checkpoint['logvar']

			# Load envs
			self.trainEnvs = checkpoint['trainEnvs']
			self.testEnvs = checkpoint['testEnvs']

			# Update seed state
			self.seed, python_seed_state, np_seed_state, torch_seed_state = checkpoint['seed_data']
			random.setstate(python_seed_state)
			np.random.set_state(np_seed_state)
			torch.set_rng_state(torch_seed_state)
		else:
			start_step = -1  # start from beginning

		# Use Adam optimizer from Pytorch, load optim state if resume
		optimizer = torch.optim.Adam([
	  					{'params': self.mu_param, 'lr': self.mu_lr},
						{'params': self.logvar_param, 'lr': self.logvar_lr}])
		if self.decayLR['use']:
			# scheduler = torch.optim.lr_scheduler.MultiStepLR(
									# optimizer, 
									# milestones=self.decayLR['milestones'], 
									# gamma=self.decayLR['gamma'])
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=self.decayLR['gamma'], patience=10, threshold=1e-3, threshold_mode='rel')
		if self.saved_model_path is not "":
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		# Determine how many policies for one env
		if self.use_antithetic:
			num_trial_per_env = 2*self.num_epsilon
		else:
			num_trial_per_env = self.num_epsilon

		# Extract env configs
		obj_pos_all, obj_orn_all, obj_path_ind_all, obj_path_list_all = self.trainEnvs

		# Repeat env config for policies in one env
		obj_pos_all = np.tile(obj_pos_all, (num_trial_per_env,1))
		obj_orn_all = np.tile(obj_orn_all, (num_trial_per_env,1))
		obj_path_ind_all = np.tile(obj_path_ind_all, (num_trial_per_env))
		obj_path_list_all = obj_path_list_all*num_trial_per_env

		# Run steps
		for step in range(self.numSteps):
			if step <= start_step:
				continue

			step_start_time = time.time()
			with torch.no_grad():  # speed up
				# Make a copy for the step
				mu_ps = self.mu_param.clone().detach()
				logvar_ps = self.logvar_param.clone().detach()
				mu_pr = self.mu_pr.clone()
				logvar_pr = self.mu_pr.clone()

				# Sample xi used for the step
				if self.use_antithetic:
					epsilons = torch.normal(mean=0., std=1., 
									size=(self.numTrainEnvs*self.num_epsilon, self.z_total_dim))
					epsilons = torch.cat((epsilons, -epsilons))  # antithetic
				else:
					epsilons = torch.normal(mean=0., std=1., 
								size=(self.numTrainEnvs*self.num_epsilon, 
              							self.z_total_dim))
				sigma_ps = (0.5*logvar_ps).exp()
				zs_all = mu_ps + sigma_ps*epsilons

				# Run trials without GUI
				success_list = self.rollout_env.parallel(
							zs_all=zs_all, 
			 				objPos=obj_pos_all, 
				 			objOrn=obj_orn_all,
							objPathInd=obj_path_ind_all,
					  		objPathList=obj_path_list_all)
				cost_env = torch.tensor([1-s for s in success_list]).float()
				emp_rate = np.mean(success_list)

				# Include PAC-Bayes reg in ES
				theta = zs_all
				kld, R = self.get_pac_bayes(
								self.numTrainEnvs, 
			  					self.delta, 
				   				logvar_ps, 
					   			logvar_pr, 
						  		mu_ps,
								mu_pr)
				reg = np.sqrt(R)
				log_pt_pr = torch.sum(
						0.5*(logvar_pr-logvar_ps) + \
						(theta-mu_pr)**2/(2*logvar_pr.exp()) - \
				 		(theta-mu_ps)**2/(2*logvar_ps.exp()) \
							, dim=1)

				# Get cost, check if include PAC-Bayes cost
				if self.include_reg:
					cost_es = cost_env + log_pt_pr/(4*self.numTrainEnvs*reg)
				else:
					cost_es = cost_env

				# Get epsilons from mu and zs
				grad_mu, grad_logvar = compute_grad_ES(
        						cost_es-torch.mean(cost_es), 
              					epsilons, 
                   				sigma_ps,
                       			method=self.ES_method)

			# Print and record result
			reg = reg.item()
			cost_env = torch.mean(cost_env).item()
			bound = 1-cost_env-reg
			print("\n", step, "Emp:", emp_rate, "Env:", cost_env, "Reg:", reg, "Bound:", bound, "KL:", kld)
			print('mu:', self.mu_param.data)
			print('logvar:', self.logvar_param.data)
			print('Time: %s\n' % (time.time() - step_start_time))

			# Save mu and logvar if at best McAllester bound
			if bound > self.best_bound_data[1]:
				self.best_bound_data = (emp_rate, bound, step, mu_ps, logvar_ps, (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()))
			if emp_rate > self.best_emp_data[0]:
				self.best_emp_data = (emp_rate, bound, step, mu_ps, logvar_ps, (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()))

			# Save training details, cover at each step
			self.cost_env_his += [cost_env]
			self.reg_his += [reg]
			self.kl_his += [kld]
			self.lr_his += [optimizer.state_dict()['param_groups'][0]['lr']] # only lr for mu since for sigma would be the same
			torch.save({
				'training_his':(self.cost_env_his, self.reg_his, self.kl_his, self.lr_his),
				'cur_data': (mu_ps, logvar_ps),
				'best_bound_data': self.best_bound_data,
				'best_emp_data': self.best_emp_data,
				'seed_data':(self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()),
				'actor_pr_path':self.actor_pr_path,
				'json_data':self.json_data,
			}, self.result_path+'train_details')  # not saving optim_state, grad

			# Do not update params until after saving results
			self.mu_param.grad = grad_mu
			self.logvar_param.grad = grad_logvar
			optimizer.step()

			# Decay learning rate if specified
			if self.decayLR['use']:
				scheduler.step(emp_rate)

			# Save model every 5 epochs
			if step % 5 == 0 and step > 0:
				torch.save({
						'step': step,
						'mu': self.mu_param,
						"logvar": self.logvar_param,
						'optimizer_state_dict': optimizer.state_dict(),
						"cost_env_his": self.cost_env_his,
						"reg_his": self.reg_his,
						"kl_his": self.kl_his,
						"lr_his": self.lr_his,
						'best_bound_data': self.best_bound_data,
						'best_emp_data': self.best_emp_data,
						"trainEnvs": self.trainEnvs,
						"testEnvs": self.testEnvs,
						"seed_data": (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()),
						}, self.model_path+'model_'+str(step))


	def estimate_train_cost(self, mu_ps, logvar_ps):
		# Extract envs
		objPos, objOrn, objPathInd, objPathList = self.trainEnvs

		# Run training trials
		estimate_success_list = []
		for sample_ind in range(self.L):
			with torch.no_grad():  # speed up
				print('\nRunning sample %d out of %d...\n' % (sample_ind+1, self.L))

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
		return np.mean(array([1-s for s in estimate_success_list]))


	def estimate_true_cost(self, mu_ps, logvar_ps):
		# Extract envs
		objPos, objOrn, objPathInd, objPathList = self.testEnvs

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
		return np.mean(array([1-s for s in estimate_success_list]))


	def compute_final_bound(self, best_data):

		# Retrive mu and logvar from best bound step, or best emp step
		step_used = best_data[2]
		mu_ps = best_data[3]
		logvar_ps = best_data[4]
		seed, python_seed_state, np_seed_state, torch_seed_state = best_data[5]
		mu_pr = self.mu_pr.detach()  # prior, checked all zeros
		logvar_pr = self.logvar_pr.detach()  # prior, checked all zeros

		# Reload seed state
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		random.setstate(python_seed_state)
		np.random.set_state(np_seed_state)
		torch.set_rng_state(torch_seed_state)

		# Get estimated true cost using test envs
		print('Estimating true cost...')
		true_estimate_cost = self.estimate_true_cost(mu_ps, logvar_ps)

		# Get estimated train cost using trian envs and L=100
		print('Estimating training cost (may take a while)...')
		train_estimate_start_time = time.time()
		train_estimate_cost = self.estimate_train_cost(mu_ps, logvar_ps)
		print('\n\n\nTime to run estimate training cost:', time.time()-train_estimate_start_time)

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

		return step_used, R, maurer_bound, quad_bound, inv_bound, train_estimate_cost, true_estimate_cost


	def get_pac_bayes(self, N, delta, logvar_ps, logvar_pr, mu_ps, mu_pr):
		kld = (-0.5*torch.sum(1 \
							+ logvar_ps-logvar_pr \
							-(mu_ps-mu_pr)**2/logvar_pr.exp() \
							-(logvar_ps-logvar_pr).exp())
		 		).item()  # as scalar
		R = (kld + np.log(2*np.sqrt(N)/delta))/(2*N)
		return kld, R  # as scalar, not tensor


if __name__ == '__main__':

	# Read JSON config
	json_file_name = sys.argv[1]

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
	trainer = TrainGrasp_PAC_ES(
	 				json_file_name=json_file_name, 
		 			result_path=result_path,
      				model_path=model_path)

	# Train
	trainer.train()

	# Get bounds using best bound step, save
	step_used, R, maurer_bound, quad_bound, inv_bound, train_estimate_cost, true_estimate_cost= trainer.compute_final_bound(trainer.best_bound_data)
	print('Using best bound, step', step_used)
	print('R:', R)
	print("Maurer Bound:", maurer_bound)
	print("Quadratic Bound:", quad_bound)
	print("KL-inv bound:", inv_bound)
	print("Train estimate:", 1-train_estimate_cost)
	print("True estimate:", 1-true_estimate_cost)
	print('\n')
	np.savez(result_path+'bounds_best_bound.npz',
		step=step_used,
		R=R,
		maurer_bound=maurer_bound,
		quad_bound=quad_bound,
		inv_bound=inv_bound,
		train_estimate_cost=train_estimate_cost,
		true_estimate_cost=true_estimate_cost,
		)

	# Get bounds using best empirical rate step, save
	step_used, R, maurer_bound, quad_bound, inv_bound, train_estimate_cost, true_estimate_cost= trainer.compute_final_bound(trainer.best_emp_data)
	print('Using best emp, step', step_used)
	print('R:', R)
	print("Maurer Bound:", maurer_bound)
	print("Quadratic Bound:", quad_bound)
	print("KL-inv bound:", inv_bound)
	print("Train estimate:", 1-train_estimate_cost)
	print("True estimate:", 1-true_estimate_cost)
	print('\n')
	np.savez(result_path+'bounds_best_emp.npz',
		step=step_used,
		R=R,
		maurer_bound=maurer_bound,
		quad_bound=quad_bound,
		inv_bound=inv_bound,
		train_estimate_cost=train_estimate_cost,
		true_estimate_cost=true_estimate_cost,
		)
