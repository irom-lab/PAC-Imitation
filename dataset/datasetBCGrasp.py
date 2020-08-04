import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import warnings
import os
warnings.filterwarnings('ignore')


class train_dataset(Dataset):
	'''
	This class reads data of the raw image files reducing the RAM requirement;
	tradeoff is slow speed.
	'''    
	def __init__(self, trainTrialsList, dataFolderName, device):
		self.device = device
		self.trainTrialsList = trainTrialsList
		self.dataFolderName = dataFolderName
		self.len = len(trainTrialsList)

	def __getitem__(self, index):
		graspInd = self.trainTrialsList[index]
		data = np.load(self.dataFolderName+str(graspInd)+'.npz')

		return (torch.from_numpy(data['depth']).float().to(self.device),
				torch.from_numpy(data['eePos']).float().to(self.device),
				torch.from_numpy(data['yaw_enc']).float().to(self.device),
				)

	def getBatch(self, indexList):
		# Always push to GPU
		depth_batch = torch.empty((0,128,128))
		# eePos_batch = torch.empty((0,3))

		for graspInd in indexList:
			data = np.load(self.dataFolderName+str(graspInd)+'.npz')
			depth = torch.from_numpy(data['depth']).float().unsqueeze(0)
			depth_batch = torch.cat((depth_batch, depth))

		return depth_batch.to('cuda:0')

	def __len__(self):
		return self.len


class test_dataset(Dataset):
	'''
	This class reads data of the raw image files reducing the RAM requirement;
	tradeoff is slow speed.    
	'''    
	def __init__(self, testTrialsList, dataFolderName, device):
		self.device = device
		self.testTrialsList = testTrialsList
		self.dataFolderName = dataFolderName
		self.len = len(testTrialsList)

	def __getitem__(self, index):
		graspInd = self.testTrialsList[index]
		data = np.load(self.dataFolderName+str(graspInd)+'.npz')

		return (torch.from_numpy(data['depth']).float().to(self.device),
				torch.from_numpy(data['eePos']).float().to(self.device),
				torch.from_numpy(data['yaw_enc']).float().to(self.device),
				)

	def __len__(self):
		return self.len
