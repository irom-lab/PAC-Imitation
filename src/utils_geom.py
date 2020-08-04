import sys
import numpy as np
from numpy import array


def NearZero(z):
	"""
	Determines whether a scalar is small enough to be treated as zero
	:param z: A scalar input to check
	:return: True if z is close to zero, false otherwise
	Example Input:
		z = -1e-7
	Output:
		True
	"""
	return abs(z) < 1e-6


##########################################################################
######### Conversion between S03 and euler angle, quaternion #########
##########################################################################

def euler2rot(a):
	"""
	Generate rotation matrix from euler angles using intrinsic ZYZ convention
	Reference: MLS Chap.2 Euler Angles

	INPUTS
	  a - (3,) - euler angles around intrinsic z,y,z axes
	OUTPUTS
	  R - (3,3) - rotation matrix as Rz * Ry * Rz
	"""

	a = np.asarray(a).flatten()

	# Trig functions of rotations
	c = np.cos(a[0:3])
	s = np.sin(a[0:3])

	# Rotation matrix as Rz * Ry * Rz
	R = np.array([
		[c[0]*c[1]*c[2]-s[0]*s[2], -c[0]*c[1]*s[2]-s[0]*c[2], c[0]*s[1]],
		[s[0]*c[1]*c[2]+c[0]*s[2], -s[0]*c[1]*s[2]+c[0]*c[2], s[0]*s[1]],
		[-s[1]*c[2], s[1]*s[2], c[1]]
		])
	return R


def rot2euler(R, zyx=False):
	"""
	Generate euler angles from rotation matrix using intrinsic ZYZ/ZYX convention
	Reference: MLS Chap.2 Euler Angles

	INPUTS
	  R - (3,3) - rotation matrix SO(3)
	OUTPUTS
	  a - (3,) - euler angles, i.e. rotations around z,y,z axes
	"""

	if zyx:
		yaw = np.arctan2(R[1,0], R[0,0])
		pitch = np.arcsin(R[2,0])
		roll = np.arctan2(R[2,1], R[2,2])
		return np.array([yaw, pitch, roll])
	else:
		beta = np.arctan2(np.sqrt(R[2,0]**2+R[2,1]**2), R[2,2])
		alpha = np.arctan2(R[1,2]/np.sin(beta), R[0,2]/np.sin(beta))
		gamma = np.arctan2(R[2,1]/np.sin(beta), -R[2,0]/np.sin(beta))
		return np.array([alpha, beta, gamma])


def quat2rot(q, w_first=False):
	"""
	Generate rotation matrix from quaternion

	INPUTS
	  q - (4,) - (unnormalized) quaternion, w_first - x,y,z,w
	OUTPUTS
	  R - (3,3) - rotation matrix
	"""

	if w_first:
		a = q[1]
		b = q[2]
		c = q[3]
		w = q[0]
	else:
		a = q[0]
		b = q[1]
		c = q[2]
		w = q[3]

	R = np.zeros((3,3))
	R[0,0] = 1-2*b**2-2*c**2
	R[0,1] = 2*a*b-2*c*w
	R[0,2] = 2*a*c+2*b*w
	R[1,0] = 2*a*b+2*c*w
	R[1,1] = 1-2*a**2-2*c**2
	R[1,2] = 2*b*c-2*a*w
	R[2,0] = 2*a*c-2*b*w
	R[2,1] = 2*b*c+2*a*w
	R[2,2] = 1-2*a**2-2*b**2

	return R


def rot2quat(R):
	"""
	Generate quaternion from rotation matrix

	INPUTS
	  R - (3,3) - rotation matrix
	OUTPUTS
	  q - (4,) - (unnormalized) quaternion, x,y,z,w
	"""

	w = 0.5*np.sqrt(1+np.trace(R))
	if w < 1e-12:
		w = 1e-12
	a = 0.25/w*(R[2,1]-R[1,2])
	b = 0.25/w*(R[0,2]-R[2,0])
	c = 0.25/w*(R[1,0]-R[0,1])
	return np.array([a,b,c,w])


def euler2quat(a):
	"""
	Generate quaternion from Euler angle, intrinsic ZYZ

	INPUTS
	  q - (4,) - (unnormalized) quaternion, x,y,z,w
	OUTPUTS
	  a - (3,) - Euler angles
	"""

	return rot2quat(euler2rot(a))


def quat2euler(q, zyx=False):
	"""
	Generate Euler angle, intrinsic ZYZ, from quaternion, or intrinsic ZYX

	INPUTS
	  a - (3,) - Euler angles
	OUTPUTS
	  q - (4,) - (unnormalized) quaternion, x,y,z,w
	"""
	return rot2euler(quat2rot(q), zyx)


def quat2aa(q):
	q = np.asarray(q)
	axis = q[:3]/np.linalg.norm(q[:3])
	angle = 2*np.arctan2(np.linalg.norm(q[:3]), q[3])
	return axis, angle


def rot2aa(R):
	angle = np.arccos((np.trace(R)-1)/2)
	axis = array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*np.sin(angle))
	return axis, angle


def log_rot(R):
	"""
	Generate angular velocity so(3) from SO(3)
	"""

	t = np.clip(np.trace(R), -0.999, 0.999)
	theta = np.arccos((t-1)/2)
	return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*np.sin(theta))


def adjoint(pos, quat):
	"""

	Following MLS convention, wrench = [force, torque]
	Modern Robotics uses the opposite, thus the skew and zeros 3x3 are opposite

	"""

	rot = quat2rot(quat)

	return np.vstack((
		# np.hstack((rot, skew(pos)@rot)),
		np.hstack((rot, skew(pos).dot(rot))),
		np.hstack((np.zeros((3,3)), rot))
	))


def homogeneous(pos, quat):
	rot = quat2rot(quat)

	return np.vstack((
		np.hstack((rot, np.asarray(pos).reshape(3,1))),
		np.array([[0,0,0,1]])
	))


############################################################################
################## Conversion involving normal vector #####################
############################################################################


def vec2rot(x,y):
	"""
	Find SO(3) that rotate x vector to y, not unique since not aligning frames but just normals
	https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
	anti-parallel case: http://en.citizendium.org/wiki/Rotation_matrix
	"""
	x = np.asarray(x).flatten()
	x = x/np.linalg.norm(x)
	y = np.asarray(y).flatten()
	y = y/np.linalg.norm(y)

	if np.linalg.norm(x + y) < 1e-4: # anti-parallel
		if (abs(x[2])-1) < 1e-2: # f_z = +-1
			return np.array([[-1,0,0],
							 [0,1,0],
							 [0,0,-1]])
		else:
			# print(x)
			# print(y)
			# print(np.array([[-(x[0]**2-x[1]**2), -2*x[0]*x[1], 0],
			#                  [-2*x[0]*x[1], (x[0]**2-x[1]**2), 0],
			#                  [0, 0, -(1-x[2]**2)]])/(1-x[2]**2))
			return np.array([[-(x[0]**2-x[1]**2), -2*x[0]*x[1], 0],
							 [-2*x[0]*x[1], (x[0]**2-x[1]**2), 0],
							 [0, 0, -(1-x[2]**2)]])/(1-x[2]**2)
	else: # Rodriguez
		v = np.cross(x,y)
		s = np.linalg.norm(v)
		c = np.dot(x,y)
		vs = skew(v)
		return np.eye(3) + vs + vs.dot(vs)*(1/(1+c))


def orient(z):
	"""
	R = orient  rotation matrix bringing vector in line with [0,0,1]
	INPUTS
	  z - 3 x 1 - vector to align with [0,0,1]
	OUTPUTS
	  R - 3 x 3 - rogation matrix which orients coordinate system
	"""
	x0 = z.reshape((3,1))
	R1 = euler(np.array([0,np.arctan2(x0[0,0],x0[2,0]),0]))
	x1 = np.dot(R1,x0)
	R2 = euler(np.array([-np.arctan2(x1[1,0],x1[2,0]),0,0]))
	x2 = np.dot(R2,x1)

	return np.dot(R2,R1)


def vecQuat2vec(v,q):
	"""
	Rotate a vector v by a quaternion q (a,b,c,w), return a 3D vector
	"""
	r = np.concatenate((v,[0]))  # add zero to the end of the array
	q_conj = np.array([-q[0],-q[1],-q[2],q[3]])
	out = quatMult(quatMult(np.array(q),r),q_conj)[:3]
	return out/np.linalg.norm(out)


def vec2quat(x,y):
	"""
	Find quaternion that rotates x vector to y , not unique since not aligning frames but just normals
	Reference: https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
	"""
	out = np.zeros(4)
	out[:3] = np.cross(x, y)
	out[3] = np.linalg.norm(x)*np.linalg.norm(y)+np.dot(x, y)
	if np.linalg.norm(out) < 1e-4:
		return np.append(-x, [0])  # 180 rotation
	return out/np.linalg.norm(out)


################################################################################


def quatDist(p, q):

	"""
	Find the distance between two quaternions.
	Reference: http://www.boris-belousov.net/2016/12/01/quat-dist/
	"""

	p = np.asarray(p)
	q = np.asarray(q)
	p /= np.linalg.norm(p)
	q /= np.linalg.norm(q)

	quatIP = np.dot(np.array(p), np.array(q))
	return np.arccos(2*quatIP**2 - 1)


def quatMult(p, q):
	"""
	Multiply two quaternions (a,b,c,w)
	"""
 
	w = p[3]*q[3] - np.dot(p[:3], q[:3])
	abc = p[3]*q[:3] + q[3]*p[:3] + np.cross(p[:3], q[:3])
	return np.hstack((abc, w))


def quatInverse(p):
	"""
	Assume (a,b,c,w). Inverse of quaternion is to negate the vector
	"""
	p = array(p)
	return np.hstack((-p[:3], p[3]))


def skew(z):
	"""
	Convert 3D vector to 3x3 skew-symmetric matrix
	"""
	return np.array([[0,    -z[2], z[1]],
					[z[2],  0,    -z[0]],
					[-z[1], z[0], 0]])


def angleBwVec(p,q):
	"""
	Get angle between two vectors
	"""
	p = np.array(p)
	q = np.array(q)
	ct = np.dot(p,q)/(np.linalg.norm(p)*np.linalg.norm(q))
	return np.arccos(ct)


def SO3_6D_np(b1, a2):
	b2 = a2 - np.dot(b1, a2)*b1
	b2 /= np.linalg.norm(b2)
	b3 = np.cross(b1, b2)
	return b2, b3


# def wrap2pi(angle):
# 	if angle > np.pi:
# 		angle -= 2*np.pi
# 	elif angle < -np.pi:
# 		angle += 2*np.pi
# 	return angle

################################################################################


def QuinticTimeScaling(Tf, t):
	"""Computes s(t) for a quintic time scaling
	:param Tf: Total time of the motion in seconds from rest to rest
	:param t: The current time t satisfying 0 < t < Tf
	:return: The path parameter s(t) corresponding to a fifth-order
			 polynomial motion that begins and ends at zero velocity and zero
			 acceleration
	Example Input:
		Tf = 2
		t = 0.6
	Output:
		0.16308
	"""
	return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
		   + 6 * (1.0 * t / Tf) ** 5


def LinearTimeScaling(Tf, t):
	"""
	Computes s(t) for a quintic time scaling
	"""
	return t/Tf
