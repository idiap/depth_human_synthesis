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

"""Implements 3D character class to extract annotations."""


import bpy
import math
import mathutils
import os
import numpy as npy
#import cv2


colors = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]


def create_material(objId, matName, color):
  """Creates a material to assign to meshes of a 3D character.

  Args:
    objId: New provided Id for the material.
    matName: Material name.
    color: color to assign to the materials.

  Returns:
    Blender object material.
  """
  print('[INFO] Creating material with ID', objId)
  mat = bpy.data.materials.new(name=matName)
  bpy.data.materials[matName].pass_index = objId
  mat.diffuse_color = color
  mat.diffuse_shader = 'LAMBERT'
  mat.use_shadeless = True
  mat.diffuse_intensity = 1.0
  mat.specular_color = (1,1,1)
  mat.specular_shader = 'COOKTORR'
  mat.specular_intensity = 0.5
  mat.alpha = 1.0
  mat.ambient = 1

  return mat


class Model():
  """Class to handle 3D human character motion simulation and data extraction."""
  def __init__(self, key, params, id_=1):
    """Initializes 3D character and Mocap sequence.

    Args:
      key: 3D character name key.
      params: Parameters of 3D character model file, mocap file and position.
      id_: 3D character id in the rendering scene.
    """
    self.key = key
    self.modelFile = params[0]
    self.mocapFile = params[1]
    self.position = params[2]
    self.id = id_
    self.materialColor = [0,0,1]
    print("[INFO] (%s) The index is %d and key %s" %\
        (self.__class__.__name__, self.id, self.key))

  def get_id(self):
    """Geet model id in the scene."""
    return self.id

  def get_color(self):
    """Get material color of 3D character."""
    return self.materialColor
      
  def loadAndRetarget(self):
    """Loads a Makehuman model and do retargeting of a Mocap sequence."""
    # Mhx is the model type that is optimized for CMU mocap data
    # The inmediate loaded object is set to context.object. This way
    # the name of the model can be retrived
    bpy.ops.import_scene.makehuman_mhx2(filepath=self.modelFile)
          
    # self.actorName = bpy.context.object.name
    ## Select the ARMATURE and set it as the active object in the scene
    # bpy.context.scene.objects.active = bpy.data.objects[self.actorName]
    bpy.context.scene.objects.active = bpy.data.objects[self.key]

    bpy.ops.object.mode_set(mode='POSE')
    # Perform the retargeting using the mocap datafile
    bpy.ops.mcp.load_and_retarget(filepath=self.mocapFile)

    # Add material to the meshes and set the material index
    self.matName = self.key + "_complete_mat"
    mat = create_material(self.id, self.matName, colors[self.id])
    self.materialColor =  colors[self.id]

    # TODO add index id only to the meshes of the object
    objs = bpy.context.scene.objects
    for key in objs.keys():
      if self.key == key:
        continue
      elif self.key in key:
        print('[INFO] Matching keys', self.key, key)
        objs[key].pass_index = 1.0 # self.key

        if objs[key].data.materials:
          print('[INFO] setting material!')
          objs[key].data.materials[0] = mat
        else:
          print('[INFO] setting material!')
          objs[key].data.materials.append(mat)

  def moveToTargetPoint(self):
    """Translate the 3D character to the fixed target point."""
    translation = self.position - self.getModelPosition()
    obj = bpy.data.objects[self.key]
    obj.location += mathutils.Vector([translation.x, translation.y, 0])                     
    bpy.data.objects[self.key].select = True
    bpy.context.scene.objects.active = bpy.data.objects[self.key]
    action =  bpy.context.object.animation_data.action  
    action.fcurves[4].mute = True
    action.fcurves[5].mute = True
    action.fcurves[6].mute = True
  
  def getModelPosition(self):
    """Returns the 3D character position."""
    # World coordinates
    leftShoulderPos = mathutils.Vector(self.getBonePosition('LeftShoulder'))
    rightShoulderPos = mathutils.Vector(self.getBonePosition('RightShoulder'))
    return (leftShoulderPos + rightShoulderPos) / 2
      
  def getBonePosition(self, boneKey):
    """Get the world position of bone accessed by boneKey.

    Args:
      boneKey: String of the name of the bone to be accessed.

    Return: 
      Bone position's vector in world coordinates.
    """
    # bpy.ops.object.mode_set(mode='POSE')
    vFil = mathutils.Vector([0,0,0,1])

    person = bpy.data.objects[self.key]
    bone = person.pose.bones[boneKey]

    boneLoc = bone.matrix * vFil
    wmtx = bpy.data.objects[self.key].matrix_world       
        
    boneLoc = wmtx * mathutils.Vector(boneLoc[0:3])    
    return boneLoc

  def rotateModel(self, angle):               
    """Applies rotation to the 3D characeter.

    Args:
      angle: Angle in radians to apply the rotation to the model.
    """
    bpy.data.objects[self.key].select = True
    bpy.context.scene.objects.active = bpy.data.objects[self.key]
    action =  bpy.context.object.animation_data.action  
    # Disable animations curves linked to rotation
    action.fcurves[0].mute = True
    action.fcurves[1].mute = True
    action.fcurves[2].mute = True
    action.fcurves[3].mute = True
    bpy.ops.transform.rotate(value=angle, axis=(0, 0, 1))   

  def getStartAndEndFrame(self):
    """Obtain the number of first and last frame of the mocap sequence.
    Returns:
      A tuple of size 2 with the number of first and last frames.
    """
    bpy.data.objects[self.key].select = True
    bpy.context.scene.objects.active = bpy.data.objects[self.key]
    action =  bpy.context.object.animation_data.action 
    return (int(action.frame_range[0]), int(action.frame_range[1]))           

  def getBoneData(self):
    """Extracts the locations of the 3D character bones in world coordinates.

    Returns:
      A list with dictionaries for each bone. Each dictionary contains: name of 
      the bone, bone location in 3D world coordinates, bone matrix, and head and
      tail bone 3D world coordinates.
    """
    # World coordinates
    boneData = []     
    obj = bpy.data.objects[self.key]         
    wmtx = bpy.data.objects[self.key].matrix_world

    print('[INFO] Get bone data')
    print(wmtx)

    # for bone in bpy.context.scene.objects.active.pose.bones:       
    for bone in obj.pose.bones:        
      #boneLoc = mathutils.Vector(self.getBonePosition(bone.name))
      # get Bone Position gives the vector multiplied by the matrix_world of object
      boneLoc = self.getBonePosition(bone.name)
      boneLoc = mathutils.Vector(boneLoc)
      # bone.tail and bone.head need to be transformed by matrix_world
      boneTail = wmtx * mathutils.Vector(bone.tail)
      boneHead = wmtx * mathutils.Vector(bone.head)
      boneData.append({'name': bone.name,
                       'boneLoc': boneLoc,
                       'boneMatrix': bone.matrix,
                       'boneTail': boneTail,
                       'boneHead': boneHead})

    return boneData             

  def get_eyes_points(self):
    """Extracts the coordinates of the eyes of the 3D character.

    Returns:
      Two vectors containing the eye coordinates, one for each eye.
    """
    # Get the centers of the eyes meshes
    eye1, eye2 = None, None

    # In order to get the transformed meshes we have to se the scene in context
    scene = bpy.context.scene
    matrixWorld = bpy.data.objects[self.key].matrix_world
    print('[INFO] The model key is', self.key, bpy.data.objects[self.key].mode)

    eyesKey = self.key + ':High-poly'

    print('[INFO] Extracting eyes positions.')
    print(matrixWorld)
    eyes = None
    for key in bpy.data.objects.keys():
      print(key)
      if eyesKey in key:
          eyes = bpy.data.objects[key]
          print('[INFO] Found eyes meshes with objKey', key)
          break

    eyesMeshes = eyes.to_mesh(scene, True, 'PREVIEW')

    vertices = eyesMeshes.vertices
    data = npy.zeros(shape=(len(vertices),3))

    print('[INFO] The size of vertices ',len(vertices), eyes.mode)

    for i in range(data.shape[0]):
      vertex = vertices[i].co
      data[i,0] = vertex[0]
      data[i,1] = vertex[1]
      data[i,2] = vertex[2]

    #print(data)
    data = data.astype(npy.float32)

    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #flags = cv2.KMEANS_RANDOM_CENTERS
    #compactness, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, flags)

    eye1 = [data[0,0], data[0,1], data[0,2], 1.0]
    eye2 = [data[data.shape[0]-1,0], data[data.shape[0]-1,1], data[data.shape[0]-1,2], 1.0]
    eye1 = mathutils.Vector(eye1)
    eye2 = mathutils.Vector(eye2)

    print(eye1, eye2)
    eye1 = matrixWorld*eye1
    eye2 = matrixWorld*eye2
    eye1 = mathutils.Vector(eye1[0:3])
    eye2 = mathutils.Vector(eye2[0:3])

    return eye1, eye2

