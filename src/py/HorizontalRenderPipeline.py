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

"""Methods to handle the randomized rendering pipeline simulating HRI cameras.

In escence the methods handle the position of the cameras in the scene and
extracts the annotations to a XML file of the current mocap models for each of 
the placed cameras.
"""


import bpy
import math
import mathutils
import os
import numpy as npy

from CameraModule import Camera
from AnnotationManager import Frame
from AnnotationManager import AnnotationManager
from AnnotationManager import Person

from BaseRenderPipeline import BaseRenderPipeline
from Model import Model


class HorizontalRenderPipeline(BaseRenderPipeline):
  """Class for the management of the rendering pipeline. 

  The class is in charge of loading and retargeting a model into a Mocap 
  sequence. It also performs the extraction of the annotations frame by frame 
  and prepares the buffer for a depth render pass."""
  def __init__(self, params):
    """Constructor.

    Args:
      params: BaseRenderPipeline configuration paramters.
    """
    BaseRenderPipeline.__init__(self, params)

  def addCamera(self, cameraPosition, cameraName, lookatPoint=None):
    """ Add a new camera into the scene.

    Args:
      cameraPosition: A vector for the position of the camera.
      cameraName: Name assigned to the camera.
      lookatpoint: Vector indicating where is the camera pointing at.
    """
    scene = bpy.context.scene
    scene.render.resolution_x = self.params.cameraPars.widthpx
    scene.render.resolution_y = self.params.cameraPars.heightpx
    scene.render.resolution_percentage = self.params.cameraPars.resratio

    # setup camera sensor parameters
    self.params.cameraPars.aspectratio = \
        scene.render.pixel_aspect_x / scene.render.pixel_aspect_y      
    self.params.cameraPars.scale = self.params.cameraPars.resratio / 100.0
    self.params.cameraPars.imgAspectRatio = \
        self.params.cameraPars.widthpx / self.params.cameraPars.heightpx
    self.params.cameraPars.heightccdmm = \
        self.params.cameraPars.widthccdmm / self.params.cameraPars.imgAspectRatio

    # Create camera
    camera = Camera(self.params.cameraPars)      
    camera.lookatpoint = lookatPoint
    camera.pos = cameraPosition
    camera.computeCameraMatrix()
  
    # Create new camera object and link
    cam_data = bpy.data.cameras.new(name=cameraName)
    cam_object = bpy.data.objects.new(name=cameraName, object_data=cam_data)

    scene.objects.link(cam_object)
    scene.camera = bpy.data.objects[cam_object.name]

    camera.setBpyWorldMatrix(scene.camera.matrix_world)
    # Add remaining parameters
    scene.camera.location = cameraPosition
    scene.camera.rotation_euler = camera.generateCameraRotation()

    # set camera into blender scene and update
    bpy.data.cameras[cameraName].lens = self.params.cameraPars.lensmm
    bpy.data.cameras[cameraName].sensor_width = self.params.cameraPars.widthccdmm
    bpy.data.cameras[cameraName].sensor_height = self.params.cameraPars.heightccdmm
    bpy.data.cameras[cameraName].sensor_fit = self.params.cameraPars.sensorfit

    scene.update()
    self.depthCamera = camera

  def extractAnnotations(self):
    """Extracts annotations and saves them to an XML file."""
    annManager = AnnotationManager()
    sameOrientation = False
    
    filesId = 1
    for model in self.models:
      annManager.setFileIds(filesId, model.modelFile, model.mocapFile)
      filesId = filesId + 1
    
    # Save camera paramters into annotation file
    annManager.setCamera(self.depthCamera)

    # Get first and last frames in the Mocap sequence
    # Iterate in the range of available frames in the mocap sequence
    # and extract annotations of each frame
    self.getStartAndEndFrame()
    
    for f in range(self.startFrame, self.endFrame + 1, 1):
      # By setting the frame number we move the scene onto the state
      # of that frame in the mocap sequence.
      bpy.context.scene.frame_set(f)

      # QUESTIONMARK The model is randomly oriented as well
      # Since the models are kept at the same position we add perturbations to
      # their orientations to have different perspectives
      orientation = npy.random.vonmises(0, 4, size=None)
      for model in self.models:
        # Give a random rotation to the model
        if sameOrientation:
          model.rotateModel(orientation)
        else:    
          model.rotateModel(npy.random.vonmises(0, 4, size=None))       
      
      # Render node tree
      bpy.ops.render.render(layer='Render Layers', write_still=True)
      annManager.setFrame(self.getFrameData(f))

    annManager.writeToFile(os.path.join(self.params.annPath, 'annotations.xml'))
