import os
import numpy as np
import random


def saveURDF_Box(path, urdfName, objMass=0.1, x_dim=1, y_dim=1, z_dim=1):
    """
    # Save URDF file at the specified path with the name. Single base link.
    """

    # Write to an URDF file
    f = open(path + urdfName + '.urdf', "w+")

    f.write("<?xml version=\"1.0\" ?>\n")
    f.write("<robot name=\"%s.urdf\">\n" % urdfName)

    f.write("\t<link name=\"baseLink\">\n")
    f.write("\t\t<inertial>\n")
    f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
    f.write("\t\t\t\t<mass value=\"%.3f\"/>\n" % objMass)
    f.write("\t\t\t\t<inertia ixx=\"6e-5\" ixy=\"0\" ixz=\"0\" iyy=\"6e-5\" iyz=\"0\" izz=\"6e-5\"/>\n")
    f.write("\t\t</inertial>\n")

    f.write("\t\t<visual>\n")
    f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
    f.write("\t\t\t<geometry>\n")
    f.write("\t\t\t\t<box size=\"%.3f %.3f %.3f\"/>\n" % (x_dim, y_dim, z_dim))
    f.write("\t\t\t</geometry>\n")
    f.write("\t\t\t<material name=\"yellow\">\n")
    f.write("\t\t\t\t<color rgba=\"0.98 0.84 0.35 1\"/>\n")
    f.write("\t\t\t</material>\n")
    f.write("\t\t</visual>\n")

    f.write("\t\t<collision>\n")
    f.write("\t\t\t<origin rpy=\"0 0 0\" xyz=\"0 0 0\"/>\n")
    f.write("\t\t\t<geometry>\n")
    f.write("\t\t\t\t<box size=\"%.3f %.3f %.3f\"/>\n" % (x_dim, y_dim, z_dim))
    f.write("\t\t\t</geometry>\n")
    f.write("\t\t</collision>\n")
    f.write("\t</link>\n")
    f.write("</robot>\n")

    f.close()


def main():

	import argparse
	def collect_as(coll_type):
		class Collect_as(argparse.Action):
			def __call__(self, parser, namespace, values, options_string=None):
				setattr(namespace, self.dest, coll_type(values))
		return Collect_as
	parser = argparse.ArgumentParser(description='Box generation')
	parser.add_argument('--obj_folder', type=str)
	arg_con = parser.parse_args()
	save_obj_path = arg_con.obj_folder  # '/home/ubuntu/box/'

	# Create directory if not existed
	if not os.path.exists(save_obj_path):
		os.mkdir(save_obj_path)

	# Config all
	numObject = 3000
	x_range = [0.04,0.08] 
	y_range = [0.06,0.10]
	z_range = [0.05,0.08]
	mass_range = [0.1,0.2]
	x_dim_all = np.random.uniform(low=x_range[0], high=x_range[1], size=(numObject,))
	y_dim_all = np.random.uniform(low=y_range[0], high=y_range[1], size=(numObject,))
	z_dim_all = np.random.uniform(low=z_range[0], high=z_range[1], size=(numObject,))
	mass_all = np.random.uniform(low=mass_range[0], high=mass_range[1], size=(numObject,))

	# Generall all boxes
	for objInd in range(numObject):

		print('Generating box', objInd)

		saveURDF_Box(path=save_obj_path, 
               		 urdfName=str(objInd),
					 objMass=mass_all[objInd],
					 x_dim=x_dim_all[objInd],
					 y_dim=y_dim_all[objInd],
					 z_dim=z_dim_all[objInd],
                  	 )


if __name__ == '__main__':
	main()

