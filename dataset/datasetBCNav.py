import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import warnings
import os
warnings.filterwarnings('ignore')


def my_collate(batch):
	rgbd_list = [item[0] for item in batch]
	prim_list = [item[1] for item in batch]
	return [rgbd_list, prim_list]


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

		return (torch.from_numpy(data['rgbd']).float().to(self.device),
				torch.from_numpy(data['prim']).long().to(self.device),
				)

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

		return (torch.from_numpy(data['rgbd']).float().to(self.device),
				torch.from_numpy(data['prim']).long().to(self.device),
				)

	
	def __len__(self):
		return self.len
