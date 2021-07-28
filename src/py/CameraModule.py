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

"""Implementation of Camera module to handle camera operations."""


import numpy as npy
import math
import mathutils
import itertools


class Camera():
  """Class to handle camera parameters nd operations."""
  def __init__(self, camParams):
    """Generate the camera matrix from camera parameters.

    Args:
      camParams: Intrinsic and extrinsic camera parameters to build the 
        camera matrix.
    """
    self.par = camParams     
    self.pos = mathutils.Vector([0,0,0])
    self.lookatpoint = mathutils.Vector([0,0,0])     
    self.computeCameraMatrix()
           
  def generateCameraRotation(self):
    """Generate camera rotation matrix."""
    # Generate a direction vector from the camera position to the look at point
    lookAtVector = self.lookatpoint - self.pos
    # point the cameras '-Z' and use its 'Y' as up
    # Return a quaternion rotation from the vector and the track and up axis.
    rot_quat = lookAtVector.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    return rot_quat.to_euler()

  def verbose(self):
    """Print paramter values."""
    print("Camera name ", self.par.name)

    print("Position ", self.pos)
    print("Euler ", self.generateCameraRotation())
    print("Lookat Point ", self.lookatpoint)

    print("Camera Lenses ", self.par.lensmm)
    print("Image width ", self.par.widthpx)
    print("Image Height", self.par.heightpx)

    print("Sensor width", self.par.widthccdmm)
    print("Sensor height", self.par.heightccdmm)

    print("Scale ", self.par.scale)
    print("Image aspect ratio", self.par.aspectratio)
    print("Sensor Fit", self.par.sensorfit)

  def computeCalibrationMatrix(self):
    """ Generate calibration matrix from the camera intrinsic parameters."""
    if (self.par.sensorfit == 'VERTICAL'):
      # the sensor height is fixed (sensor fit is horizontal), 
      # the sensor width is effectively changed with the pixel aspect ratio
      s_u = self.par.widthpx * self.par.scale / self.par.widthccdmm / self.par.aspectratio
      s_v = self.par.heightpx * self.par.scale / self.par.heightccdmm
    else: # 'HORIZONTAL' and 'AUTO'
      # the sensor width is fixed (sensor fit is horizontal), 
      # the sensor height is effectively changed with the pixel aspect ratio
      s_u = self.par.widthpx * self.par.scale / self.par.widthccdmm
      s_v = self.par.heightpx * self.par.scale * self.par.aspectratio / self.par.heightccdmm

    # Parameters of intrinsic calibration matrix K
    alpha_u = self.par.lensmm * s_u
    alpha_v = self.par.lensmm * s_v
    u_0 = self.par.widthpx * self.par.scale / 2
    v_0 = self.par.heightpx * self.par.scale / 2
    skew = 0 # only use rectangular pixels

    self.K = mathutils.Matrix(((alpha_u, skew,    u_0),
                               (    0  , alpha_v, v_0),
                               (    0  , 0,        1 )))


  def computeRotationTranslationMatrix(self):
    """Computes rotation and translation matrix from the extrinsic parameters."""
    # bcam stands for blender camera
    R_bcam2cv = mathutils.Matrix(((1, 0,  0),
                                  (0, -1, 0),
                                  (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    self.R = self.generateCameraRotation().to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    self.T = -1 * self.R * self.pos

    # Build the coordinate transform matrix from world to computer vision camera
    self.R = R_bcam2cv * self.R
    # The transpose is to adjust to the oring at the top left corner of the image
    self.T = R_bcam2cv * self.T 

    # put into 3x4 matrix
    self.RT = mathutils.Matrix((self.R[0][:] + (self.T[0],),
                                self.R[1][:] + (self.T[1],),
                                self.R[2][:] + (self.T[2],)))

  def computeCameraMatrix(self):
    """Computes camera projection matrix for coordinate transformations."""
    self.computeCalibrationMatrix()
    self.computeRotationTranslationMatrix()
    self.P = self.K * self.RT

  def world2Image(self, wPoint):
    """Project point from world coordinates to image coordinate plane.

    Args:
      wPoint: Point in the 3D space in world coordinates.

    Returns:
      Point in the image coordinate system.
    """
    pImage = self.P * wPoint
    pImage /= pImage[2]
    return mathutils.Vector(pImage[0:2])              
      
  def world2Cam(self, wPoint):
    """Project from world to camera coordinates"""
    pCam = self.RT * wPoint
    # pCam = self.RT * (wPoint - self.pos) 
    return mathutils.Vector(pCam)

  def setBpyWorldMatrix(self, bpyWorldMat):
    """Set Blender world matrix"""
    self.W = bpyWorldMat

