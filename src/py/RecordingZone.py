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

"""Methods to handle recording zone traits."""

import bpy
import math
import numpy as npy

_NMAX_CAMERAS = 3


class RecordingZone():
  """Class to implement a recording zone..

  The class help to setup the recording zone to record from HRI perspectives.
  """
  class Params:
    """Parameter class."""
    def __init__(self):
        self.safeRadius = 2.0
        self.actionZoneBegin = self.safeRadius
        self.actionZoneEnd   = 6.5
        self.minHeight       = 1.0
        self.maxHeight       = 1.80

  def __init__(self, referencePoint, params):    
    """Constructor.

    Args:
      referencePoint: Reference point where to move the models to.
      params: parameters of the recording zone. 
    """
    self.referencePoint = referencePoint
    self.params = params
    self.randPars = RecordingZone.Params()

  def createCameraHorizontalPosition(self, hipsHeight, neckHeight, nbOfPositions):
    """Creates ramdom camera positions simulating HRI sensors around the person.

    The number of positions or cameras, should be less than _NMAX_CAMERAS.

    Args:
      hipsHeight: Height of the hips of a 3D character.
      neckHeight: Height of the neck.
      nbOfPositions: Number of positions to create. This is according to the
        number of 3D characters in the scene.

    Returns:
      A list of size `nbOfPositions` with the position vectors.
    """
    assert nbOfPositions<=_NMAX_CAMERAS, \
        "Maximum allowed cameras {} and provided {}".format(
            _NMAX_CAMERAS, nbOfPositions
        )

    positions = []
    angles = [npy.random.uniform(low=0.0, high=120.0),\
              npy.random.uniform(low=120.0, high=240.0),\
              npy.random.uniform(low=240.0, high=360.0)]

    activeRadius = self.randPars.actionZoneEnd - self.randPars.safeRadius

    for i in range (0, nbOfPositions):
      angle = angles[i]
      camZ = npy.random.uniform(low=self.randPars.minHeight, high=self.randPars.maxHeight)
      length = npy.random.uniform(low=0, high=activeRadius)

      camX = length * math.cos(math.radians(angle)) + self.randPars.safeRadius + self.referencePoint.x
      camY = length * math.sin(math.radians(angle)) + self.randPars.safeRadius + self.referencePoint.y

      lookatZ = npy.random.uniform(low=hipsHeight, high=neckHeight)

      # Camera position and lookatPoint pair
      position = ([camX, camY, camZ],
                  [self.referencePoint.x, self.referencePoint.y, lookatZ])
      positions.append(position)

    return positions          

  def createGround(self, clippingDist):
    """Creates a ground mesh on the floor in the shape of a square.

    Args:
      clippingDist: Distance to define the ground square dimensions."""
    mesh = bpy.data.meshes.new('ground')
    object = bpy.data.objects.new('ground', mesh)
    bpy.context.scene.objects.link(object)
    groundVerts = [
        (self.referencePoint.x + clippingDist, 
         self.referencePoint.y + clippingDist, 0),
        (self.referencePoint.x - clippingDist, 
         self.referencePoint.y + clippingDist, 0),
        (self.referencePoint.x - clippingDist,
         self.referencePoint.y - clippingDist, 0),
        (self.referencePoint.x + clippingDist, 
         self.referencePoint.y - clippingDist, 0)
    ]
    groundFace = [(0, 1, 2, 3)]
    mesh.from_pydata(groundVerts, [], groundFace)
    mesh.update()

  def createPositions(self):
    """Creates a grid of positions (x, y, z)."""       
    positions = []
    for x in self.my_range(self.params.dimX, self.params.stepX):
      for y in self.my_range(self.params.dimY, self.params.stepY):
        positions.append([
            x + self.referencePoint.x,
            y + self.referencePoint.y, 
            npy.random.uniform(low=self.params.minDimZ, high=self.params.maxDimZ)
        ])

    return positions   
           
  def my_range(self, length, step):
    """Creates a list of values between [-length / 2, +length / 2]."""
    residue = (length % step) / 2
    for x in range (0, math.trunc(length / step) + 1):
        yield x * step + residue - (length / 2)

  def createMaterial(self, materialName, transparency):
    """Creates a new material.
    
    Args:
      materialName: The name of the new material.
      transparency: The transparency of the new material, float between 0.0 
          (no transparency) and 1.0 (full transparency).
    """
    mat = bpy.data.materials.get(materialName) or bpy.data.materials.new(materialName)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # clear default nodes
    while(nodes): nodes.remove(nodes[0])
    
    output  = nodes.new('ShaderNodeOutputMaterial')
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    mix = nodes.new('ShaderNodeMixShader')

    mix.inputs[0].default_value = transparency
    
    links.new(output.inputs['Surface'], mix.outputs['Shader'])
    links.new(mix.inputs[1], diffuse.outputs['BSDF'])
    links.new(mix.inputs[2], transparent.outputs[0])
    
    return mat

  def setMaterial(self, obj, material) :
    """Sets a material to an object.
        
    Args:
        obj: The object to apply the material to.
        material: The material to be applied to the object.
    """
    if obj.data.materials:
      # assign to 1st material slot, replace existing material
      obj.data.materials[0] = material
    else:
      # no existing slot
      obj.data.materials.append(material)
    obj.active_material_index = len(obj.data.materials) - 1     
