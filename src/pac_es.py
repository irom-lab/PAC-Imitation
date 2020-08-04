import torch
import cvxpy as cvx
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def kl_inverse(q, c):
	'''Compute kl inverse using Relative Entropy Programming'''    
	p_bernoulli = cvx.Variable(2)
	q_bernoulli = np.array([q,1-q])
	constraints = [c >= cvx.sum(cvx.kl_div(q_bernoulli,p_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1, p_bernoulli[1] == 1.0-p_bernoulli[0]]
	prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)
	prob.solve(verbose=False, solver=cvx.ECOS)
	
	return p_bernoulli.value[0]


def utility(num_policy_eval):
	# Fitness rank transformation
	u = torch.zeros(num_policy_eval*2)
	for i in range(num_policy_eval*2):
		u[i] = torch.max(torch.Tensor([torch.Tensor([num_policy_eval+1]).log() - torch.Tensor([i+1]).log(), 0]))
	u /= u.sum()
	u -= 1./(2*num_policy_eval)
	return u


def compute_grad_ES(costs, epsilons, std, method='ES'):
	# Compute gradient of 1-loss

	num_policy_eval = int(costs.numel()/2) # costs for eps and -eps

	if method=='utility':
		'''Scale by utility function u instead of the loss'''
		fit_index = costs.sort().indices
		epsilons = epsilons[fit_index]

		# Fitness rank transformation
		u = torch.zeros(num_policy_eval*2)
		for i in range(num_policy_eval*2):
			u[i] = torch.max(torch.Tensor([torch.Tensor([num_policy_eval+1]).log() - torch.Tensor([i+1]).log(), 0]))
		u /= u.sum()
		u -= 1./(2*num_policy_eval)

		loss = u
		grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0]
		grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0]
		logvar = torch.log(std.pow(2))
		grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
		# The direction with the lowest cost is given the highest utility,
		# so effectively we are finding the negative of the gradient
		return -grad_mu, -grad_logvar


	if method=='ES':
		'''Pick num_fit best epsilon and scale by 1-loss function'''
		num_fit_frac = 1
		fit_index = costs.sort().indices
		num_fit = int(2*num_policy_eval*num_fit_frac)
		fit_index = fit_index[:num_fit]

		loss = costs[fit_index]
		epsilons = epsilons[fit_index]
		N = epsilons.shape[0]
		grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/N
		grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/N
		logvar = torch.log(std.pow(2))
		grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
		return grad_mu, grad_logvar


	if method=='eNES_logvar':
		'''NES'''
		num_fit_frac = 1
		fit_index = costs.sort().indices
		num_fit = int(2*num_policy_eval*num_fit_frac)
		fit_index = fit_index[:num_fit]
		if num_fit_frac < 1:
			loss = 1. - costs[fit_index]
		else:
			loss = costs[fit_index]
		epsilons = epsilons[fit_index]
		
		# Vectorized Fisher information inverse
		#   see: https://en.wikipedia.org/wiki/Fisher_information#Multivariate_normal_distribution
		F_vec_inv = torch.cat([std**2, 2*torch.ones_like(std)])
		
		grad_mu = ((torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0])
		grad_std = ((torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0])

		logvar = torch.log(std.pow(2))
		grad_logvar = 0.5 * grad_std * std

		grad_mu_NES = F_vec_inv[:std.shape[0]] * grad_mu
		grad_logvar_NES = F_vec_inv[std.shape[0]:] * grad_logvar

		return grad_mu_NES, grad_logvar_NES


	if method=='eNES_logvar_utility':
		'''NES with utility'''
		fit_index = costs.sort().indices
		epsilons = epsilons[fit_index]
		loss = utility(num_policy_eval)

		# Vectorized Fisher information inverse
		#   see: https://en.wikipedia.org/wiki/Fisher_information#Multivariate_normal_distribution
		F_vec_inv = torch.cat([std**2, 2*torch.ones_like(std)])

		grad_mu = -((torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0])
		grad_std = -((torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0])

		logvar = torch.log(std.pow(2))
		grad_logvar = 0.5 * grad_std * std

		grad_mu_NES = F_vec_inv[:std.shape[0]] * grad_mu
		grad_logvar_NES = F_vec_inv[std.shape[0]:] * grad_logvar

		return grad_mu_NES, grad_logvar_NES
