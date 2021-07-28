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

"""Methods for convertion of EXR float32 image format to uint16.

This script is normally executed by the rendering pipeline in order to save
space. All depth images are converted with a max depth valid value of 8m.
Images are converted from meters to milimeters in order to be able to save
them in uint16 format (TIFF).
"""


import os
import sys

from libtiff import TIFFimage
import cv2
import numpy as np
import scipy.io
import argparse

_MAX_DEPTH_VALUE = 8.5
_MAX_VALID_DEPTH_VALUE = 8.0


def convert_exr_to_uint16(inputPath, outputPath):
  """Converts depth images from EXR format to uint16 format.

  Args:
    inputPath: Path containing EXR depth images.
    outputPath: Path where to save uint16 depth images.
  """
  iPath = inputPath
  oPath = outputPath

  imgList = [os.path.join(iPath, x) for x in os.listdir(iPath) if x.endswith('exr')]
  imgList.sort()

  for i in range(len(imgList)):
    print('[INFO] processing %d / %d' %(i, len(imgList)))
    iName = imgList[i]
    oName = os.path.join(oPath, iName.split('/')[-1].split('.')[0])
    print('[INFO] Name of input image:', iName)
    print('[INFO] Name of output image:', oName)

    # There will be data loss for doing from floating point 
    # to uint16 data conversion
    iImg = cv2.imread(imgList[i], cv2.IMREAD_ANYDEPTH)

    print('[INFO] size of the input image %s' %(iImg.shape,))

    # Culling greater values and change from meters to milimeters 
    # as in Kinect configuration
    iImg[iImg > _MAX_VALID_DEPTH_VALUE] = _MAX_DEPTH_VALUE
    oImg = iImg * 1000.0
    oImg = np.array(oImg, dtype=np.uint16)
    print('[INFO] size of the output image %s' %(iImg.shape,))

    #cv2.imwrite(oName, oImg)
    tiff = TIFFimage(oImg, description='')
    tiff.write_file(oName, compression=None)
    #scipy.io.savemat(oName, mdict={'depth':oImg})


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Runs the convertion from hdr image to uint image of 16 bits.'
  )
  parser.add_argument('-i', '--input_path', type=str, 
      help='Path where the hdrimages are located.')
  parser.add_argument('-o', '--output_path', type=str, 
      help='Path where to save the new comverted images.')
  args = parser.parse_args()

  convert_exr_to_uint16(args.input_path, args.output_path)
