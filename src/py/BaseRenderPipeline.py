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

"""Motion retargeting and body landmark location extraction from 3D world."""


from abc import ABCMeta, abstractmethod
import bpy
import math
import mathutils
import os
import numpy as npy

from CameraModule import Camera
from AnnotationManager import Frame
from AnnotationManager import AnnotationManager
from AnnotationManager import Person

import pdb

from Model import Model

class BaseRenderPipeline:
  """Base class for the management of the rendering pipeline.

  The class is in charge of loading and retargeting a model into a Mocap 
  sequence. It also performs the extraction of the annotations frame by 
  frame and prepares the buffer for a depth render pass.
  """

  def __init__(self, params):
    """Constructor.
    Args:
      params: Paramters for motion simulation process such as models, sequence
        etc.
    """
    self.params = params
    self.depthCamera = None
    self.startFrame = 0
    self.endFrame = 0
    self.models = []
    self.createModels()

  def createModels(self):
    """Set models into stack."""
    id_= 1
    for key, value in self.params.retargetingInfos.items(): 
      self.models.append(Model(key, value, id_))
      id_ += 1

  def loadAndRetarget(self, sameOrientation):
    """Load 3D character and performs mocap sequence retargeting."""
    orientation = npy.random.vonmises(0, 4, size=None)
    for model in self.models:          
      model.loadAndRetarget()  
      # Move the retargeted model to the reference point
      model.moveToTargetPoint()
      # Give a random rotation to the model
      if sameOrientation:
        model.rotateModel(orientation)
      else:    
        model.rotateModel(npy.random.vonmises(0, 4, size=None))                

  def getStartAndEndFrame(self):
    self.startFrame = 1
    # dummy end frame, normally is less than 500
    self.endFrame = 10000
  
    for model in self.models:
      frame = model.getStartAndEndFrame()
      if frame[0] > self.startFrame:
        self.startFrame = frame[0]
      if frame[1] < self.endFrame:
        self.endFrame = frame[1]

  def reset_blend(self):
    """Delete scenes and all their objects.

    Before adding another model to perform motion simulation the Blender scene
    has to be cleaned up by removing previous model meshes, materials and 
    retargeting info.
    """
    print('[WARNING] Unlinking the objects')
    # object mode should be set in Blender to do this operation.
    bpy.ops.object.mode_set(mode='OBJECT') 

    #for scene in bpy.data.scenes:
    #    for obj in bpy.data.objects:
    #        scene.objects.unlink(obj)
    #        print('[INFO] Removing object', obj)
    #        bpy.data.objects.remove(obj)

    print('[WARNING] Deleting the objects  ')
    # select all the objects
    for obj in bpy.data.objects:
      obj.select = True

    # using blender operation to delete objects
    bpy.ops.object.delete()

    print('[WARNING] Deleting the materials  ')
    # Delete all materials
    for material in bpy.data.materials:
      material.user_clear()
      bpy.data.materials.remove(material)

#         # TODO remove this and use blender delete feature
#        for bpy_data_iter in (
#                bpy.data.objects,
#                bpy.data.meshes,
#                bpy.data.lamps,
#                bpy.data.cameras, 
#                bpy.data.armatures,
#                bpy.data.actions,
#                bpy.data.materials,
#                bpy.data.groups
#                ):
#            for id_data in bpy_data_iter:
#                bpy_data_iter.remove(id_data)
#      print('[WARNING] ***********Deleting the scenes  ')
#        for scene in bpy.data.scenes:
#            if (not (len(bpy.data.scenes) == 1)):
#                bpy.data.scenes.remove(scene)
#      print('[WARNING] *********** Finishing all the deleting  ')

          
  def getFrameData(self, frameId):
    """Extracts locations of armature landmarks in the current Mocap frame.

    Note: Normally a call to this method will take place after 
    scene.frame_set() is called.

    Args:
      frameId: Id of the frame to extract data from.

    Return: 
      A Frame object containing the list of bone's positions and their projection
      matrices.
    """           
    persons = []       
    for model in self.models:
      boneData = model.getBoneData()
      for data in boneData:
        # Transform world coordinates into image coordinates. 
        data['boneLocImg'] = self.depthCamera.world2Image(data['boneLoc'])
        data['tailLocImg'] = self.depthCamera.world2Image(data['boneTail'])
        data['headLocImg'] = self.depthCamera.world2Image(data['boneHead'])
        
        # Add camera 
        data['boneLocCam'] = self.depthCamera.world2Cam(data['boneLoc'])
        data['tailLocCam'] = self.depthCamera.world2Cam(data['boneTail'])
        data['headLocCam'] = self.depthCamera.world2Cam(data['boneHead'])

      # Adding eye information
      eye1, eye2 = model.get_eyes_points()
      #eye1 = *eye1
      #eye2 = bpy.data.objects[model.key].matrix_world*eye2
      print('[INFO] Setting eyes to bone data')
      boneEye1 = {'name': 'Eye1', 'boneLoc': eye1, 'boneLocImg': self.depthCamera.world2Image(eye1)}
      boneEye2 = {'name': 'Eye2', 'boneLoc': eye2, 'boneLocImg': self.depthCamera.world2Image(eye2)}
      boneData.append(boneEye1)
      boneData.append(boneEye2)

      personParams = Person.Params()
      personParams.personId = model.get_id()
      personParams.boneData = boneData
      personParams.materialColor = model.get_color()
      personParams.mocapFile = model.mocapFile
      personParams.modelFile = model.modelFile
      persons.append(Person(personParams))

    return Frame(frameId, persons)

  @abstractmethod
  def addCamera(self, cameraPosition, cameraName, lookatPoint=None):
    pass

  @abstractmethod
  def extractAnnotations(self):
    pass
