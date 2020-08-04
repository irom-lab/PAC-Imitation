import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import warnings
import os
warnings.filterwarnings('ignore')


def my_collate(batch):
	depth_list = [item[0] for item in batch]
	ee_list = [item[1] for item in batch]  # pose or history
	pose_action_list = [item[2] for item in batch]
	seq_len_list = [item[3] for item in batch]
	label_list = [int(item[4]) for item in batch]
	return [depth_list, ee_list, pose_action_list, seq_len_list, label_list]


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
				torch.from_numpy(data['ee_history']).float().to(self.device),
			torch.from_numpy(data['pose_action'][:,:2]).float().to(self.device),
			torch.tensor(data['num_datapoints']).long().to(self.device),
			data['label'],
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

		return (torch.from_numpy(data['depth']).float().to(self.device),
				torch.from_numpy(data['ee_history']).float().to(self.device),
			torch.from_numpy(data['pose_action'][:,:2]).float().to(self.device),
			torch.tensor(data['num_datapoints']).long().to(self.device),
			data['label'],
			)

	
	def __len__(self):
		return self.len
