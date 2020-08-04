from numpy import array
import numpy as np
import time
import torch
import ray
from src.push_rollout_env import PushRolloutEnv
from src.nn_push import PolicyNet


def main():

	import argparse
	def collect_as(coll_type):
		class Collect_as(argparse.Action):
			def __call__(self, parser, namespace, values, options_string=None):
				setattr(namespace, self.dest, coll_type(values))
		return Collect_as
	parser = argparse.ArgumentParser(description='PAC-Bayes Opt')
	parser.add_argument('--obj_folder', type=str) # '/home/ubuntu/Box_v4/'
	parser.add_argument('--posterior_path', type=str, default=None)
	arg_con = parser.parse_args()
	obj_folder = arg_con.obj_folder 
	training_details_dic_path = arg_con.posterior_path  # 'push_result/push_pac_easy/train_details'

	# Load decoder policy
	z_total_dim = 5
	actor = PolicyNet(
		input_num_chann=1,
		dim_mlp_append=10,
		num_mlp_output=2,  # x/y only
		out_cnn_dim=40,
		z_conv_dim=1,
		z_mlp_dim=4,
		img_size=150).to('cpu')
	actor.load_state_dict(torch.load('pretrained/push_pretrained_decoder.pt'))

	# Load posterior if specified path; load prior otherwise
	if training_details_dic_path is not None:
		training_details_dic = torch.load(training_details_dic_path)
		mu = training_details_dic['best_bound_data'][3]
		logvar_ps = training_details_dic['best_bound_data'][4]
		sigma = (0.5*logvar_ps).exp()
	else:
		mu = torch.zeros((1, z_total_dim))
		sigma = torch.ones((1, z_total_dim))
	print('mu:', mu)
	print('sigma:', sigma)

	# Run
	with torch.no_grad():
		# Initialize rollout env
		rollout_env = PushRolloutEnv(
							actor=actor,
							z_total_dim=z_total_dim,
							num_cpus=10,
       						y_target_range=0.15)

		#* Run single
		zs = torch.normal(mean=mu.repeat(5,1), std=sigma.repeat(5,1))
		for ind in range(5):
			info = rollout_env.roll_single(
								zs=zs[ind],
								objPos=array([0.51, -0.09, 0.035]),
								objOrn=array([0., 0., -0.6]), 
								objPath=obj_folder+'2010.urdf',
								mu=0.3,
								sigma=0.01)
			info = rollout_env.roll_single(
								zs=zs[ind],
								objPos=array([0.58, 0.012, 0.035]),
								objOrn=array([0., 0., -0.5]), 
								objPath=obj_folder+'2020.urdf',
								mu=0.3,
								sigma=0.01)
			info = rollout_env.roll_single(
								zs=zs[ind],
								objPos=array([0.62, -0.08, 0.035]),
								objOrn=array([0., 0., 0.6]), 
								objPath=obj_folder+'2030.urdf',
								mu=0.3,
								sigma=0.01)

		#* Run parallel
		# Configure all trials
		# numTrials = 200
		# obj_x = np.random.uniform(low=0.50, 
		# 							high=0.65, 
		# 							size=(numTrials, 1))
		# obj_y = np.random.uniform(low=-0.15, 
		# 						high=0.15, 
		# 						size=(numTrials, 1))
		# obj_yaw = np.random.uniform(low=-np.pi/4, 
		# 							high=np.pi/4, 
		# 							size=(numTrials, 1))
		# objPos = np.hstack((obj_x, obj_y, 0.035*np.ones((numTrials, 1))))
		# objOrn = np.hstack((np.zeros((numTrials, 2)), obj_yaw))
		# obj_ind_list = np.random.choice(a=np.arange(1000,2000), size=numTrials, replace=False)
		# objPathList = []
		# for obj_ind in obj_ind_list:
		# 	objPathList += [obj_folder + str(obj_ind) + '.urdf']
		# objPathInd = np.arange(0,len(objPathList))
		# info = rollout_env.roll_parallel(
		# 				zs_all=zs,
		# 				objPos=objPos,
		# 				objOrn=objOrn,
		# 				objPathInd=objPathInd,
		# 				objPathList=objPathList,
      	# 				mu=[0.3]*numTrials,
        #    				sigma=[0.01]*numTrials,
        #        			getTipPath=False)
		# estimate_success_list = array([s[0] for s in info])
	
		# print(estimate_success_list)
		# print('Success rate:', np.mean(estimate_success_list))
		# # print('Time used:', time.time()-start_time)


if __name__ == '__main__':
	main()