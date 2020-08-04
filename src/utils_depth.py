import math
import numpy as np
import pybullet as p


def getCameraParametersPush():
	params = {}
	params['imgW'] = 400
	params['imgH'] = 400
	params['imgW_orig'] = 1024 
	params['imgH_orig'] = 768

	# p.resetDebugVisualizerCamera(0.70, 180, -89, [0.60, 0.0, 0.0])  # 70cm away
	params['viewMatPanda'] = [-1.0, 0.0, 0.0, 0.0, 
                           	  0.0, -1.0, 0.0, 0.0, 
                              0.0, 0.0, 1.0, 0.0, 
                              0.6, 0.0, -0.7, 1.0]  # -0.7 is same as height
	params['projMatPanda'] = [1.0, 0.0, 0.0, 0.0, 
                              0.0, 1.0, 0.0, 0.0, 
                              0.0, 0.0, -1.0000200271606445, -1.0,
                              0.0, 0.0, -0.02000020071864128, 0.0]
	params['cameraUp'] = [0.0, 0.0, 1.0]
	params['camForward'] = [0.0, -0.00017464162374380976, -1.0]
	params['horizon'] = [-20000.0, -0.0, 0.0]
	params['vertical'] = [0.0, -20000.0, 3.4928321838378906]
	params['dist'] = 0.70
	params['camTarget'] = [0.6, 0.0, 0.0]

	###########################################################################

	m22 = params['projMatPanda'][10]
	m32 = params['projMatPanda'][14]
	params['near'] = 2*m32/(2*m22-2)
	params['far'] = ((m22-1.0)*params['near'])/(m22+1.0)

	return params



def getCameraParametersGrasp():
	params = {}
	params['imgW'] = 512
	params['imgH'] = 512
	params['imgW_orig'] = 1024 
	params['imgH_orig'] = 768

	# p.resetDebugVisualizerCamera(0.70, 180, -89, [0.50, 0.0, 0.0])  # 70cm away
	params['viewMatPanda'] = [-1.0, 0.0, 0.0, 0.0, 
                           	  0.0, -1.0, 0.0, 0.0, 
                              0.0, 0.0, 1.0, 0.0, 
                              0.5, 0.0, -0.7, 1.0]  # -0.7 is same as height
	params['projMatPanda'] = [1.0, 0.0, 0.0, 0.0, 
                              0.0, 1.0, 0.0, 0.0, 
                              0.0, 0.0, -1.0000200271606445, -1.0,
                              0.0, 0.0, -0.02000020071864128, 0.0]
	params['cameraUp'] = [0.0, 0.0, 1.0]
	params['camForward'] = [0.0, -0.00017464162374380976, -1.0]
	params['horizon'] = [-20000.0, -0.0, 0.0]
	params['vertical'] = [0.0, -20000.0, 3.4928321838378906]
	params['dist'] = 0.70
	params['camTarget'] = [0.5, 0.0, 0.0]

	###########################################################################

	m22 = params['projMatPanda'][10]
	m32 = params['projMatPanda'][14]
	params['near'] = 2*m32/(2*m22-2)
	params['far'] = ((m22-1.0)*params['near'])/(m22+1.0)

	return params
