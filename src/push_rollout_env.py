import time
import numpy as np
from numpy import array
import itertools
from copy import deepcopy

import pybullet as p
import ray
import torch

from src.nn_push import PolicyNet
from src.panda_env import pandaEnv
from src.utils_geom import *
from src.utils_depth import *


class PushRolloutEnv():

	def __init__(self,
				actor,
				z_total_dim,
				num_cpus=10,
				y_target_range=0.05,
				):

		# Policy for inference
		self.actor = actor
		self.z_total_dim = z_total_dim
		self.y_target_range = y_target_range

		# Initialize Panda arm environment
		self._timestep = 1./240.
		self.pandaEnv = pandaEnv(timestep=self._timestep,
								long_finger=True)

		# Camera parameters
		camera_params = getCameraParametersPush()
		self.viewMat = camera_params['viewMatPanda']
		self.projMat = camera_params['projMatPanda']
		self.width = camera_params['imgW']
		self.height = camera_params['imgH']
		self.near = camera_params['near']
		self.far = camera_params['far']

		# Object ID in PyBullet
		self._objId = None
  
		# Number of cpus for Ray, no constraint if set to zero
		self.num_cpus = num_cpus


	def shutdown(self):
		p.disconnect()


	def reset(self):
		self.pandaEnv.reset_env()


	def roll_parallel(self, zs_all, objPos, objOrn, objPathInd, objPathList, mu, sigma):
		
		numTrials = zs_all.shape[0]

		# Run in parallel
		if self.num_cpus == 0:
			ray.init()
		else:
			ray.init(num_cpus=self.num_cpus, num_gpus=0)
		info = ray.get([self.roll_parallel_wrapper.remote(self, 
									zs=zs_all[trialInd], 
									objPos=objPos[trialInd], 
									objOrn=objOrn[trialInd],
									objPath=objPathList[objPathInd[trialInd]],
									mu=mu[trialInd],
									sigma=sigma[trialInd]) for trialInd in range(numTrials)],
				 					)  # never use gui
		ray.shutdown()
		return info


	@ray.remote
	def roll_parallel_wrapper(self, zs, objPos, objOrn, objPath, mu, sigma):
		return self.push(zs, objPos, objOrn, objPath, mu, sigma, gui=False)  # never use gui

	def roll_single(self, zs, objPos, objOrn, objPath, mu, sigma):
		return self.push(zs, objPos, objOrn, objPath, mu, sigma, gui=True)

	def push(self, zs, objPos, objOrn, objPath, mu, sigma, gui):
		# Connect to an PB instance
		if gui:
			p.connect(p.GUI, options="--width=2600 --height=1800")
			p.resetDebugVisualizerCamera(0.8, 180, -45, [0.5, 0, 0])
		else:
			cid = p.connect(p.DIRECT)  # gives same id, each id is not shared among Ray instances

		######################### Reset #######################
		self.reset()

		# Load object
		if len(objOrn) == 3:  # input Euler angles
			objOrn = p.getQuaternionFromEuler(objOrn)
		self._objId = p.loadURDF(objPath, 
						   		basePosition=objPos, 
								baseOrientation=objOrn)
		
		# Change friction coeff of obj, fingers and arm
		p.changeDynamics(self._objId, -1, 
				   		lateralFriction=mu,
						spinningFriction=sigma, 
						frictionAnchor=1)  # mass specified in URDF
		self.pandaEnv.change_friction_coeffs(mu, sigma)

		# Let the object settle
		for _ in range(20):
			p.stepSimulation()

		# Set arm to initial pose
		initial_ee_pos = array([0.35, 0.0, 0.18])
		initial_ee_orn = array([0.966003, 0.0002059, 0.2585298, 0.0007693])  # 60 deg
		self.pandaEnv.reset_arm_joints_ik(initial_ee_pos, 
							   			  initial_ee_orn, 
								  	 	  fingerPos=0.015)

		# Time configuration
		useRealTime = 0
		p.setRealTimeSimulation(useRealTime)
		cur_timestep = 0
		real_hz = 5  # control hz running on real system
		action_timesteps = int(240/real_hz)  # convert to PyBullet

		# Scaling
		action_scale = array([50, 50, 20])  #x/y/yaw
		eePose_scale = array([3, 6, 3])  #x/y/yaw

		# Initialize reward and states
		states = np.zeros((10))
		reward = 0
		success = 0

		while cur_timestep < 3000:

			# Get observations, convert to NN input
			depth, states = self.get_obs(eePose_scale, states)
			depth_input = torch.from_numpy(depth).float().to('cpu').unsqueeze(0).unsqueeze(0)  # 1x1x200x200
			states_input=torch.from_numpy(states).float().to('cpu').unsqueeze(0)

			# Infer action using the model
			action_pred = self.actor(img=depth_input, 
								zs=zs.reshape(1, -1), 
								mlp_append=states_input).squeeze(0).numpy()

			# Apply scaling to action, no gradient in rollout so fint
			action_pred /= action_scale[:2]
			posAction = np.hstack((action_pred[:2], 0))
			eulerAction = [0,0,0]

			# Get finger tips
			tipPos, tipQuat = self.pandaEnv.get_gripper_tip_long()

			# Execute action for 48 timesteps
			cur_timestep, _ = self.pandaEnv.move_pos(
										relative_pos=posAction,
										relative_global_euler=eulerAction,
										gripper_target_pos=0.015,# fixed opening
										numSteps=action_timesteps, 
										timeStep=cur_timestep)

			# Check cost, quit if success or robot goes crazy
			reward = self.check_reward()
			success = self.check_success()
			if tipPos[0] > 0.75 or abs(tipPos[1] > 0.30) or success:
				break

		# Close instance and return result
		if gui:
			time.sleep(1)
		p.disconnect()
		return success, reward


	def check_success(self):
		# Discrete success
		objPos, _ = p.getBasePositionAndOrientation(self._objId)
		if objPos[0] > 0.75 and abs(objPos[1]) < self.y_target_range:
			return 1
		return 0


	def check_reward(self):
		# Continuous reward
		objPos, _ = p.getBasePositionAndOrientation(self._objId)
		if objPos[0] < 0.75:
			dist = np.sqrt(np.sum((objPos[:2] - array([0.75,0.0]))**2))
			return max(0.7 - dist, 0) 
		else:
			return 1-max(abs(objPos[1])-self.y_target_range, 0)*5  # full reward within 15cm, =0.5 if at y=25cm


	def get_obj(self):
		objPos, objOrn = p.getBasePositionAndOrientation(self._objId)
		return array(objPos), array(objOrn)


	def get_obs(self, eePose_scale, ee_history):
		depth = self.get_depth()
		depth = ((0.7 - depth)/0.12).clip(max=1)
		depth += np.random.normal(0,0.0005, depth.shape)
		depth[depth >= 0.99] = 0.0

		eePos, eeOrn = self.pandaEnv.get_ee()
		eePos -= array([0.35,0.0,0.18])
		eePose = np.hstack((eePos[:2], 0))  # zero yaw
		eePose *= eePose_scale

		# Update history
		ee_history[2:10] = ee_history[0:8]
		ee_history[0:2] = eePose[:2]

		return depth, ee_history


	def get_depth(self):
		img_arr = p.getCameraImage(width=self.width, 
							 	   height=self.height, 
								   viewMatrix=self.viewMat, 
								   projectionMatrix=self.projMat, 
								   flags=p.ER_NO_SEGMENTATION_MASK)
		depth = np.reshape(img_arr[3], (400,400))[(200-75):(200+75), (200-75):(200+75)]  # 150x150
		depth = self.far*self.near/(self.far - (self.far - self.near)*depth)
		return depth
