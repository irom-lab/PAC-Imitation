import sys
from gibson2.envs.base_env import BaseEnv
import time
import pybullet as p
import numpy as np
from numpy import array
import torch
import matplotlib.pyplot as plt
import ray
import itertools
import cv2
import copy
from importlib import reload  
import math

from src.nn_nav import CNN_nav, Decoder_nav
from src.utils_geom import euler2quat

tracking_camera = {
	'yaw': 270,
	'z_offset': 1,
	'distance': 1.5,
	'pitch': -20
}


def quat2aa(q):
	q = np.asarray(q)
	axis = q[:3]/np.linalg.norm(q[:3])
	angle = 2*np.arctan2(np.linalg.norm(q[:3]), q[3])
	return axis, angle


class NavRolloutEnv():

	def __init__(self,
				CNN,
				decoder,
				z_dim,
				num_cpus=10,
				num_gpus=4,
				verbose=False,
				step_length=0.20,
				max_rollout_steps=100,
				start_pos=[0.5, 0.0, 0.0],
				start_orn=list(euler2quat([0,0,np.pi*7/8])),
				batch_size=5,
				AWS=0,
				collision_thres=-0.03,
				config_version=0, # new env
				):

		# Policy for inference
		self.CNN = CNN
		self.decoder = decoder
		self.z_dim = z_dim
		self.num_cpus = num_cpus
		self.AWS = AWS
		self.batch_size = batch_size
		if not AWS:
			self.num_gpus = 1
		else:
			self.num_gpus = num_gpus

		# Config for all trials (no global gibson env)
		self.verbose = verbose
		self.step_length = step_length
		self.max_rollout_steps = max_rollout_steps
		self.start_pos = start_pos
		self.start_orn = start_orn
		self.collision_thres = collision_thres
	
		self.target_pos = [-4.3,1.7,0.9]
		self.target_orn = euler2quat([0,0,-np.pi/4])
		self.target_thres = 0.8		
		self.target_path = '/home/ubuntu/YCB/003_cracker_box/google_16k/textured.obj'
		self.gui_config = 'config/fetch_p2p_nav_gui_v'+str(config_version)+'.yaml'
		self.parallel_config = 'config/fetch_p2p_nav_parallel_v'+str(config_version)+'.yaml'


	def roll_parallel(self, zs_all, obj_poses_all, obj_paths_all):

		# Default settings, can be changed in the last iteration
		num_trials = zs_all.shape[0]
		num_batch = self.num_cpus
		batch_size = self.batch_size
		numIters = math.ceil(num_trials/num_batch/batch_size)
		numTrialPerIter = num_batch*batch_size

		success_list_all = []
		for iter_ind in range(numIters):

			# Check how many trials left
			num_trials_left = num_trials - iter_ind*numTrialPerIter
			
			# Determine how many batches for this iter. Make sure to split evenly when there is not many trials left
			iter_batch_size = min(batch_size, math.ceil(num_trials_left/num_batch))

			# Count actual number of batches for this iteration
			num_batch_iter = 0

			# Split data for batch
			zs_batch_all = []
			obj_poses_batch_all = []
			obj_paths_batch_all = []
			for batch_ind in range(num_batch):
				batch_start_ind = batch_ind*iter_batch_size + iter_ind*numTrialPerIter
				batch_end_ind = (batch_ind+1)*iter_batch_size + iter_ind*numTrialPerIter

				# Check start index is over limit
				if batch_start_ind >= num_trials:
					break
 
				# Count actual number of batches
				num_batch_iter += 1
 
				# Check end limit
				batch_end_ind = min(num_trials, batch_end_ind)
 
				# Add to batch
				zs_batch_all += [zs_all[batch_start_ind:batch_end_ind]]
				obj_poses_batch_all +=[obj_poses_all[batch_start_ind:batch_end_ind]]
				obj_paths_batch_all +=[obj_paths_all[batch_start_ind:batch_end_ind]]

			if self.num_cpus == 0:
				ray.init()
			else:
				ray.init(num_cpus=self.num_cpus, num_gpus=0)

			if self.AWS:
				gpu_ids = list(np.arange(self.num_gpus))*(math.ceil(num_batch_iter/self.num_gpus))
			else:
				gpu_ids = [0]*num_batch_iter

			# Run actual number of batches
			success_list = ray.get([self.parallel_wrapper.remote(self, 
								zs_all=zs_batch_all[batchInd], 
								obj_poses_all=obj_poses_batch_all[batchInd], 
								obj_paths_all=obj_paths_batch_all[batchInd],
								gpu_id=gpu_ids[batchInd])
							for batchInd in range(num_batch_iter)])
			for batch_ind in range(num_batch_iter):
				success_list_all += success_list[batch_ind]

			# Shutdown and restart Ray for each iteration
			ray.shutdown()

		return array(success_list_all)


	@ray.remote
	def parallel_wrapper(self, zs_all, obj_poses_all, obj_paths_all, gpu_id, get_robot_pos=False):
		mode = 'headless'
		return self.nav(zs_all, obj_poses_all, obj_paths_all, gpu_id, mode, get_robot_pos)


	def roll_single(self, zs, obj_poses, obj_paths, gpu_id=0, mode='pbgui'):
		return self.nav(zs, [obj_poses], [obj_paths], gpu_id, mode)[0]


	def nav(self, zs_all, obj_poses_all, obj_paths_all, gpu_id, mode, get_robot_pos=False):

		# Choose config file based on gui or parallel
		if mode == 'pbgui':
			config_file = self.gui_config
		else:
			config_file = self.parallel_config

		# Initialize gibson env
		env = BaseEnv(config_file=config_file, mode=mode, verbose=False,device_idx=gpu_id)
		renderer = env.simulator.renderer
		robot = env.robots[0]
		robot_id = robot.robot_ids[0]
		mesh_body_id = env.simulator.scene.mesh_body_id
		pb_id = env.simulator.cid  # PyBullet

		robot_pos_all = []
		success_list = []
		numTrials = len(obj_poses_all)

		# Figure out zs, make sure NxBx(z_dim)
		if len(zs_all.shape) == 2:  # repeat for steps
			zs_all = zs_all.unsqueeze(1).repeat(1,self.max_rollout_steps,1)
		if zs_all.shape[0] == 1:  # repeat for all trials
			zs_all = zs_all.repeat(numTrials,1,1)

		# Use same target for all trials, load in PB in gui mode (no need to collision)
		if mode == 'pbgui':
			target_visual_id = p.createVisualShape(
				p.GEOM_MESH,
				fileName=self.target_path,
				meshScale=2,
			)
			pb_target_id = p.createMultiBody(
				baseVisualShapeIndex=target_visual_id,
				basePosition=self.target_pos,
				baseOrientation=self.target_orn,
			)
		renderer.load_object(
			obj_path=self.target_path,
			transform_pos=self.target_pos,
			transform_orn=self.target_orn,
			load_texture=True,
			scale=2,
		)
		renderer.add_instance(object_id=len(renderer.visual_objects)-1)

		# Run all trials
		for trial_ind in range(numTrials):

			robot_pos_trial = np.empty((0,3))

			# Extract for trial
			obj_poses = obj_poses_all[trial_ind]
			obj_paths = obj_paths_all[trial_ind]

			# Reset robot
			p.resetBasePositionAndOrientation(robot_id, self.start_pos, self.start_orn)	

			# Load objects, load texture in PB in gui mode
			num_objects = len(obj_paths)
			shapenet_obj_ids = []
			pb_obj_ids = []
			for (i, (pose, path)) in enumerate(zip(obj_poses, obj_paths)):

				pos = [pose[0], pose[1], 0.0]
				orn = euler2quat([0,0,pose[2]])

				pb_obj_id = []
				visual_id = -1  # if in parallel
				if mode == 'pbgui':
					visual_id = p.createVisualShape(
						p.GEOM_MESH,
						fileName=path,
						meshScale=1.,
					)
				collision_id = p.createCollisionShape(
					p.GEOM_MESH,
					fileName=path,
					meshScale=1.,
				)
				pb_obj_id = p.createMultiBody(
					baseVisualShapeIndex=visual_id,
					baseCollisionShapeIndex=collision_id,
					basePosition=pos,
					baseOrientation=orn,
				)
				pb_obj_ids.append(pb_obj_id)

				# Load object in iGibson renderer
				renderer.load_object(
					obj_path=path,
					transform_pos=pos,
					transform_orn=orn,
					load_texture=True
				)
				vis_obj_id = len(renderer.visual_objects) - 1
				shapenet_obj_ids.append(vis_obj_id)
				renderer.add_instance(object_id=shapenet_obj_ids[-1])

			# Run steps
			success = 0
			past_prim = [0,0,0,0]
			for step_ind in range(self.max_rollout_steps):

				# Get robot position
				pos = robot.get_position()
				if get_robot_pos:
					robot_pos_trial = np.concatenate((robot_pos_trial, array(pos).reshape(1,3)))

				# Move the PyBullet camera with the robot, only in GUI
				if mode == 'pbgui':
					pos[2] += tracking_camera['z_offset']
					dist = tracking_camera['distance']/robot.scale
					axis, angle = quat2aa(robot.get_orientation())	# compute robot's yaw angle to adjust the camera
					if axis[2] < 0:
						angle = 2*np.pi - angle
					p.resetDebugVisualizerCamera(dist, tracking_camera['yaw']+angle*180/np.pi, tracking_camera['pitch'], array(pos))

				# Get camera views
				rgb = env.simulator.renderer.render_robot_cameras(modes=('rgb'))[0]  # 200x200x3
				depth = env.simulator.renderer.render_robot_cameras(modes=('3d'))[0][:,:,2]  # 200x200x1
				if mode == 'pbgui':
					self.vis_robot_view(rgb, depth, sensors=['rgb','norm_depth'])
				if self.robot_collision(robot, mesh_body_id, pb_obj_ids):
					if mode == 'pbgui':
						print("Collision!")
					break

				# Process images
				rgb = rgb[:,:,:3]
				depth = (-depth/6.0).clip(min=0.0, max=1.0)
				depth[depth==0] = 1  # fill infinity
				depth = depth[:,:,np.newaxis]

				# Infer primitive
				rgbd_input = torch.from_numpy(np.concatenate((rgb, depth), axis=2)).float().to('cpu').unsqueeze(0).permute(0,3,1,2)  # 1x4x200x200
				zs_input = zs_all[trial_ind, step_ind].reshape(1,self.z_dim)
				img_feat = self.CNN(rgbd_input)
				prim = self.decoder(img_feat, zs_input).squeeze(0)
				prim = torch.argmax(prim)
	
				# Update prim history
				past_prim[1:4] = past_prim[0:3]

				# Check if stuck bt left/right, move forward if so
				if any(ele == 2 for ele in past_prim) and \
					any(ele == 3 for ele in past_prim) and \
			   		all(ele != 0 for ele in past_prim) and \
					all(ele != 1 for ele in past_prim):
					prim = 0

				# Check if stuck bt forward/backward, force back (until turn left/right by prim_pred)
				if any(ele == 0 for ele in past_prim) and \
					any(ele == 1 for ele in past_prim) and \
			   		all(ele != 2 for ele in past_prim) and \
					all(ele != 3 for ele in past_prim):
					if pos[1] > 0.5:
						prim = 2  # turn left if on right side
					else:
						prim = 3
    
				# Update history for current
				past_prim[0] = prim

				# Execute primitive
				if prim == 0:
					robot.move_forward(forward=self.step_length)
				elif prim == 1:		
					robot.move_backward(backward=self.step_length)
				elif prim == 2:
					robot.turn_left(delta=self.step_length)
				elif prim == 3:
					robot.turn_right(delta=self.step_length)

				# Debug
				if mode == 'pbgui':
					time.sleep(0.1)

				# Check distance to target
				dist = np.sqrt((pos[0]-self.target_pos[0])**2+\
							   (pos[1]-self.target_pos[1])**2)

				if self.verbose:
					print("Frame: %d, Dist to target: %.3f" % (step_ind, dist))
				
				# End if close enough
				if dist < self.target_thres:
					success = 1  # keep trial
					break

			for pb_obj_id in pb_obj_ids:
				p.removeBody(pb_obj_id)
			renderer.instances = renderer.instances[:3]# leave wall/robot/target
			renderer.textures = renderer.textures[:7]

			# Add to batch success
			success_list += [success]
			if get_robot_pos:
				robot_pos_all += [robot_pos_trial]

		# Close image windows if in gui
		if mode == 'pbgui':
			cv2.destroyAllWindows()
			print('Steps used:', step_ind+1)

		# Disconnect PB, clean renderer
		env.clean()

		if get_robot_pos:
			return success_list, robot_pos_all
		else:
			return success_list


	def robot_collision(self, robot, mesh_body_id, pb_obj_ids):
		collision_scene = list(p.getClosestPoints(
			bodyA=robot.robot_ids[0], 
			bodyB=mesh_body_id, 
			distance=0.001)  # small threshold
		)
		# check collision position on mesh, count as collision if higher than 5cm (filter out the floor, assume floor as z=0)
		collision_scene_count = len([col[6][2] for col in collision_scene if col[6][2] > 0.05])
		collision_shapenet = []
		for shapenet_obj_id in pb_obj_ids:
			collision_shapenet.extend(
				list(p.getClosestPoints(
				bodyA=robot.robot_ids[0], 
				bodyB=shapenet_obj_id, 
				distance=self.collision_thres))  # small threshold
			)
		collision_shapenet_count = len([col for col in collision_shapenet if col[8] < self.collision_thres]) # check if negative distance (penetration)
		if collision_shapenet_count > 0 or collision_scene_count > 0:
			return True
		else:
			return False


	def vis_robot_view(self, rgb=[], depth=[], sensors=['rgb','depth','norm_depth']):
		'''Visualize the robot's vision'''
		for sensor in sensors:
			if sensor=='rgb':
				cv2.imshow('RGB', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
				cv2.waitKey(1)
			elif sensor=='depth':
				cv2.imshow('Depth', depth)
				cv2.waitKey(1)
			elif sensor=='norm_depth':
				norm_depth = cv2.normalize(
					depth, None, alpha=0, beta=1, 
					norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
				)
				cv2.imshow('Normalized Depth', norm_depth)
				cv2.waitKey(1)
			else:
				print("Warning: "+sensor+" is not a valid sensor display option")
