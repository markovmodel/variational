# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 10:55:24 2015

@author: fvitalini
"""

"""
This script contains examples of usage for the classes:
    RamachandranBasis
    RamachandranProductBasis
which are contained in  the variational package.
"""


import variational
import numpy as np

#Use of the function RamachandranBasis

from variational.basissets.ramachandran import RamachandranBasis
alabasis = RamachandranBasis('A', radians=False) #load the residue centered basis
                                                 #function for residue Alanine and
                                                 #default force field (ff_AMBER99SB_ILDN)
                                                 #three eigenvectors are considered (order=2)
                                                 #expects the timeseries in degrees.     
atraj = np.load('torsion_A.npy') #the file contains the phi/psi timeseries for residue A
print atraj[0:10,:] #first 10 timesteps only
ala_basis_traj=alabasis.map(atraj) # projects the trajectory onto the residue basis function
print ala_basis_traj[0:10, :] #first 10 timesteps only


#Use of the function RamachandranProductBasis

# 1: Different number excitations
from variational.basissets.ramachandran import RamachandranProductBasis
FGAILbasis=RamachandranProductBasis('FGAIL', n_excite=3, radians=False) #load the residue centered basis
                                                                         #functions for residues F-G-A-I-L and
                                                                         #default force field (ff_AMBER99SB_ILDN)
                                                                         #three eigenvectors are considered (order=2)
                                                                         #up to 3 excited residue per basis function (n_excite=3)
                                                                         #expects the timeseries in degrees.     
FGAIL_traj = np.load('torsion_FGAIL.npy') #the file contains the phi/psi timeseries for residues FGAIL
print FGAIL_traj[0:10,:] #first 10 timesteps only
FGAIL_basis_set_traj, FGAIL_basis_set_list=FGAILbasis.map(FGAIL_traj) #projects the trajectory onto the residue basis functions
print FGAIL_basis_set_list
print FGAIL_basis_set_traj[0:10,:] #first 10 timesteps only

# 2: Select only residues FG
FGbasis=RamachandranProductBasis('FGAIL',include_res=[True,True,False,False,False], radians=False) #load the residue centered basis
                                                                                                    #functions for residues F-G and
                                                                                                    #default force field (ff_AMBER99SB_ILDN)
                                                                                                    #three eigenvectors are considered (order=2)
                                                                                                    #2 excited residue per basis function (n_excite=2)
                                                                                                    #expects the timeseries in degrees.     
FG_basis_set_traj, FG_basis_set_list=FGbasis.map(FGAIL_traj) #projects the trajectory onto the residue basis functions
print FG_basis_set_list
print FG_basis_set_traj[0:10,:] #first 10 timesteps only
print FG_basis_set_traj[0:10,0] #first 10 timesteps of basis function 00
print FG_basis_set_traj[0:10,1] #first 10 timesteps of basis function 01
print FG_basis_set_traj[0:10,8] #first 10 timesteps of basis function 22