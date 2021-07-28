###############################################################################
# DepthHuman: A tool for depth image synthesis for human pose estimation.
# 
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Angel Martinez <angel.martinez@idiap.ch>,
# Melanie Huck <melanie.huck@idiap.ch>
# 
# This file is part of 
# DepthHuman: A tool for depth image synthesis for human pose estimation.
# 
# DepthHuman is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# DepthHuman is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with DepthHuman. If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""Utils and tools for rendering pipeline."""


import os
import sys
import bpy
import math
import mathutils

#import cv2
import numpy as np
import scipy.io


def convert_exr_to_uint16(inputPath, outputPath):
  iPath = inputPath
  oPath = outputPath

  imgList = [os.path.join(iPath, x) for x in os.listdir(iPath) if x.endswith('exr')]
  imgList.sort()

  for i in range(len(imgList)):
    print('[INFO] processing %d / %d' %(i, len(imgList)))
    iName = imgList[i]
    oName = os.path.join(oPath, iName.split('/')[-1].split('.')[0] + '.mat')
    print('[INFO] Name of input image:', iName)
    print('[INFO] Name of output image:', oName)
    """
    # There will be data loss for doing from floating point to uint16 data conversion
    iImg = cv2.imread(imgList[i], cv2.IMREAD_ANYDEPTH)

    print('[INFO] size of the input image %s' %(iImg.shape,))

    iImg[iImg > 8.0] = 8.5
    oImg = iImg * 1000.0
    oImg = np.array(oImg, dtype=np.uint16)
    print('[INFO] size of the output image %s' %(iImg.shape,))

    #cv2.imwrite(oName, oImg)
    #tiff = TIFFimage(oImg, description='')
    #tiff.write_file(oName, compression=None)
    scipy.io.savemat(oName, mdict={'depth':oImg})
    """


def get_3x4_RT_matrix_from_blender(cam):
  """Convert projection matrix to computer vision matrix."""
  # bcam stands for blender camera
  R_bcam2cv = mathutils.Matrix(
      ((1, 0,  0),
       (0, -1, 0),
       (0, 0, -1)))

  # Transpose since the rotation is object rotation, 
  # and we want coordinate rotation
  # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
  # T_world2bcam = -1 * R_world2bcam * location
  #
  # Use matrix_world instead to account for all constraints
  location, rotation = cam.matrix_world.decompose()[0:2]
  R_world2bcam = rotation.to_matrix().transposed()

  # Convert camera location to translation vector used in coordinate changes
  # T_world2bcam = -1*R_world2bcam*cam.location
  # Use location from matrix_world to account for constraints:     
  T_world2bcam = -1 * R_world2bcam * location

  # Build the coordinate transform matrix from world to computer vision camera
  R_world2cv = R_bcam2cv * R_world2bcam
  T_world2cv = R_bcam2cv * T_world2bcam

  # put into 3x4 matrix
  RT = mathutils.Matrix((
      R_world2cv[0][:] + (T_world2cv[0],),
      R_world2cv[1][:] + (T_world2cv[1],),
      R_world2cv[2][:] + (T_world2cv[2],)
       ))

  return RT


def fun_messages(text='Hello World', mode='header'):
  """Shows a message according to the input mode."""
  # Print line.
  if mode=='line': print('******************************************')

  # Print header.
  if mode=='header':
    print('\n******************************************')
    print('{}'.format(text))
    print('Unicity Project')
    print('Idiap Research Institute')
    print('Switzerland - 2017')
    print('******************************************\n')

  # Print title.
  if mode=='title':
    print('******************************************')
    print('{}'.format(text))
    print('******************************************')

  # Print process.
  if mode=='process': print('[Process] {}'.format(text))
  # Print information.
  if mode=='info': print('[Info] {}'.format(text))
  # Print warning.
  if mode=='warning': print('[Warning] {} :|'.format(text))
  # Print warning.
  if mode=='question': print('[Question] {}'.format(text))
  # Print error.
  if mode=='error':
      print('[Error] {} :('.format(text))
      raise


# Check file.
def fun_check_file(path):
  """Checks if the input file path exists."""

  # Check file path.
  if not os.path.exists(path): fun_messages('File path not found', 'error')

# Check directory.
def fun_check_directory(path):
  """Checks if the input directory path exists."""

  # Check directory path.
  if not os.path.isdir(path): fun_messages('Directory path not found %s' %path , \
                                           'error')
