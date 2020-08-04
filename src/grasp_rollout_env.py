import sys
import time
import pybullet as p
import numpy as np
from numpy import array
import torch
import matplotlib.pyplot as plt
import ray
import itertools

from src.utils_depth import *
from src.nn_grasp import PolicyNet
from src.panda_env import pandaEnv
from src.utils_geom import quatMult, euler2quat, quat2euler


class GraspRolloutEnv():

	def __init__(self,
				actor,
				z_total_dim,
				num_cpus=10,
				checkPalmContact=True,
				useLongFinger=False,
				):

		# Policy for inference
		self.actor = actor
		self.z_total_dim = z_total_dim

		# Initialize Panda arm environment
		self.env = pandaEnv(long_finger=useLongFinger)
		self.checkPalmContact = checkPalmContact  # if contact -> fail

		# Camera parameters
		camera_params = getCameraParametersGrasp()
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
		self.env.reset_env()


	def parallel(self, zs_all, objPos, objOrn, objPathInd, objPathList):
		numTrials = zs_all.shape[0]
		if self.num_cpus == 0:
			ray.init()
		else:
			ray.init(num_cpus=self.num_cpus, num_gpus=0)
		success_list = ray.get([self.parallel_wrapper.remote(self, 
                                    zs=zs_all[trialInd], 
                                    objPos=objPos[trialInd], 
                                    objOrn=objOrn[trialInd], 
                                    objPath=objPathList[objPathInd[trialInd]]) 
                        			for trialInd in range(numTrials)])
		ray.shutdown()
		return success_list


	@ray.remote
	def parallel_wrapper(self, zs, objPos, objOrn, objPath):
		return self.grasp(zs, objPos, objOrn, objPath, gui=False)


	def single(self, zs, objPos, objOrn, objPath, gui=True, save_figure=True, figure_path=None, mu=None, sigma=None):
		return self.grasp(zs, objPos, objOrn, objPath, gui, save_figure, figure_path, mu, sigma)


	def grasp(self, zs, objPos, objOrn, objPath, gui, save_figure=False, figure_path=None, mu=None, sigma=None):

		# Connect to an PB instance
		if gui:
			p.connect(p.GUI, options="--width=2600 --height=1800")
			p.resetDebugVisualizerCamera(0.8, 180, -45, [0.5, 0, 0])
		else:
			p.connect(p.DIRECT)

		zs = zs.reshape(1, -1)

		######################### Reset #######################
		self.reset()

		# Load object
		if len(objOrn) == 3:  # input Euler angles
			objOrn = p.getQuaternionFromEuler(objOrn)
		self._objId = p.loadURDF(objPath, 
                           		basePosition=objPos, 
                             	baseOrientation=objOrn)
		p.changeDynamics(self._objId, -1, 
                   		lateralFriction=self.env._mu,
                     	spinningFriction=self.env._sigma, 
                      	frictionAnchor=1, 
                       	mass=0.1)

		# Let the object settle
		for _ in range(20):
			p.stepSimulation()

		######################### Decision #######################

		initial_ee_pos_before_depth = array([0.3, -0.5, 0.35])
		initial_ee_orn = array([1.0, 0.0, 0.0, 0.0])

		# Set arm to a pose away from camera image, keep fingers open
		self.env.reset_arm_joints_ik(initial_ee_pos_before_depth, 
                               		initial_ee_orn, 
                                	fingerPos=0.04)

		# Get observations
		depth = ((0.7 - self.get_depth())/0.20).clip(max=1.0)
		if gui:
			plt.imshow(depth, cmap='Greys', interpolation='nearest')
			plt.show()
		depth = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)

		# Infer action
		pred = self.actor(depth, zs).squeeze(0).detach().numpy()
		target_pos = pred[:3]
		target_pos[:2] /= 20
		target_pos[0] += 0.5  # add offset
		target_yaw = np.arctan2(pred[3], pred[4])
		if target_yaw < -np.pi/2:
			target_yaw += np.pi
		elif target_yaw > np.pi/2:
			target_yaw -= np.pi

		######################## Motion ##########################

		# Reset to target pos above
		target_pos_above = target_pos + array([0.,0.,0.05])
		target_euler = [-1.57, 3.14, 1.57-target_yaw]
		for _ in range(3):
			self.env.reset_arm_joints_ik(target_pos_above, euler2quat(target_euler), fingerPos=0.04)
		self.env.grasp(targetVel=0.10)  # keep finger open before grasping

		# Reach down to target pos, check if hits the object
		_, success = self.env.move_pos(absolute_pos=target_pos,
                                 	   absolute_global_euler=target_euler,
                                       numSteps=150, 
                                    #    checkContact=self.checkContact,
                                       checkPalmContact=self.checkPalmContact, 
                                       objId=self._objId)
		if not success:
			p.disconnect()
			return success

		# Grasp
		self.env.grasp(targetVel=-0.10)
		self.env.move_pos(absolute_pos=target_pos, 
                		  absolute_global_euler=target_euler, 
                    	  numSteps=100)

		# Lift
		self.env.move_pos(absolute_pos=target_pos_above, 
                    	  absolute_global_euler=target_euler, 
                          numSteps=150)

		# Check success
		table_mug_contact = p.getContactPoints(self._objId, 
												self.env._tableId, 
												linkIndexA=-1, 
												linkIndexB=-1)
		table_mug_contact = [i for i in table_mug_contact if i[9] > 1e-3]
		if len(table_mug_contact) == 0:
			success = 1
		else:
			success = 0

		# Close instance and return result
		p.disconnect()
		return success


	def get_depth(self):
		img_arr = p.getCameraImage(width=self.width, 
							 	   height=self.height, 
								   viewMatrix=self.viewMat, 
								   projectionMatrix=self.projMat, 
								   flags=p.ER_NO_SEGMENTATION_MASK)
		depth = np.reshape(img_arr[3], (512,512))[192:192+128, 192:192+128]  # center 128x128 from 512x512
		depth = self.far*self.near/(self.far - (self.far - self.near)*depth)
		return depth
