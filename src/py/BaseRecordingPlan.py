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

"""Methods to handle the setup of Blender to execute recording plans."""


from abc import ABCMeta, abstractmethod
import os
import numpy as npy
import mathutils
import shutil
import xml.etree.ElementTree as ET
import bpy 
import random
import pdb
import math
from itertools import product

from HorizontalRenderPipeline import HorizontalRenderPipeline
from CameraModule import Camera

colors = [(0,0,0, 1), (1,0,0,1), (0,1,0,1), (0,0,1,1)]


class BaseRecordingManager:
  """Base class to setup Blender and manage recording scene for image synthesis.

  The class setups the Nodes for setup the rendering process and access to
  depth buffers.
  """
  def __init__(self, configParams, recParams, nModels=1):
    """Initialization of recording manager.

    Args:
      configParmas: Configuration parameters from the main configuration file.
      recParams: Recording parameters form recording manager configuration file.
      nModels: Number of models.
    """
    self.configParams = configParams
    self.referencePointParams = recParams[0]
    self.airlockParams = recParams[1]
    self.camParamsList = recParams[2]
    self.nModelsInScene = nModels
    print('[INFO] (%s) Number of models in the scene %d' %\
        (self.__class__.__name__, nModels))

  def prepareRenderNodes(self):
    """Prepare and connect the rendering nodes in Blender.

    Get ready for rendering the z buffer. The current method is based on 
    connecting rendering nodes for a render pass of z buffer. The method 
    peprares the nodes to save 8bit image and floating point image.
    """ 
    scene = bpy.context.scene
    
    if not scene.render.engine == 'CYCLES' :
      scene.render.engine = 'CYCLES'
    
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    # Clear default nodes
    nodes.clear()

    render_layers = nodes.new('CompositorNodeRLayers')
    scene.render.layers['RenderLayer'].use_pass_object_index = True # Set pass index usage to render obj mask
    scene.render.layers['RenderLayer'].use_pass_material_index = True # Set pass index for materials
    scene.render.layers['RenderLayer'].use_pass_z = True

    # Add output node for openexr file: HDR image that contains depth in meters
    output_exr = nodes.new('CompositorNodeOutputFile')
    output_exr.format.file_format = 'OPEN_EXR'
    output_exr.format.use_zbuffer = True
    output_exr.format.exr_codec = 'NONE'
    output_exr.format.color_depth = '32' # Save as floating point numbers
    output_exr.label = 'hdrNode'
   
    output_png = nodes.new('CompositorNodeOutputFile')
    output_png.format.file_format = 'PNG'
    output_png.format.use_zbuffer = True
    output_png.format.color_mode = 'BW'
    output_png.format.color_depth = '8'
    output_png.format.compression = 0
    output_png.label = 'uint8Node'

    # Add node for material output
    output_mat = nodes.new('CompositorNodeOutputFile')
    output_mat.format.file_format = 'PNG'
    output_mat.format.color_mode = 'RGB'
    output_mat.format.color_depth = '8'
    output_mat.format.compression = 0
    output_mat.label = 'material'
   
    # Add node to normalize and see images as Depths
    # To visualize depth as image (0-255 normalized)
    normalize_node = nodes.new('CompositorNodeNormalize')
    
    # Link all the nodes as in nodes section in the blender GUI
    scene.node_tree.links.new(render_layers.outputs['Depth'], 
                            normalize_node.inputs['Value'])
                                                            
    scene.node_tree.links.new(normalize_node.outputs['Value'], 
                            output_png.inputs['Image'])

    # Needs no normalization since this is the depth in meters
    scene.node_tree.links.new(render_layers.outputs['Depth'], 
                            output_exr.inputs['Image'])

    # Add color ramp
    ramp_node = nodes.new('CompositorNodeValToRGB')
    ramp_norm_node = nodes.new('CompositorNodeMath')
    ramp_norm_node.operation = 'DIVIDE'
    ramp_norm_node.inputs[1].default_value = 2.0
    ramp_node.color_ramp.elements.new(0.0)

    scene.node_tree.links.new(render_layers.outputs['IndexMA'],
                              ramp_norm_node.inputs[0])
    scene.node_tree.links.new(ramp_norm_node.outputs['Value'],
                              ramp_node.inputs['Fac'])
    scene.node_tree.links.new(ramp_node.outputs['Image'],
                              output_mat.inputs['Image'])

    for i in range(self.nModelsInScene):
      element = ramp_node.color_ramp.elements[i+1]
      element.color = colors[i+1]
      element.position = 0.5*(i+1.0)
      print(element.color, element.position)


    # Add output mask for actor in the scene
    output_mask = nodes.new('CompositorNodeOutputFile')
    output_mask.format.file_format = 'PNG'
    output_mask.format.use_zbuffer = False
    output_mask.format.color_mode = 'BW'
    output_mask.format.color_depth = '8'
    output_mask.format.compression = 0
    output_mask.label = 'maskNode'

    # Add ID mask node and add index 
    # TODO add several mask nodes for each actor in the scene
    idmask_node = nodes.new('CompositorNodeIDMask')
    idmask_node.index = 1.0
    idmask_node.use_antialiasing = True

    scene.node_tree.links.new(render_layers.outputs['IndexOB'],
                              idmask_node.inputs['ID value'])
    scene.node_tree.links.new(idmask_node.outputs['Alpha'], 
                              output_mask.inputs['Image'])
  
  def setNodePaths(self, hdrPath, uint8Path, maskPath, materialPath):
    """Creates and set the paths where the rendered images will be saved.
    
    Args: 
      hdrPath: Path to the location where the hdr image will be saved.
      uint8Path: Path to the location where the png image will be saved.
      maskPath: Path to the location to save binary mask.
      materialPath: Path to the location to save colored mask (material images)
    """
    nodes = bpy.context.scene.node_tree.nodes
    for node in nodes:
      if node.label == 'hdrNode':
        node.base_path = hdrPath
      elif node.label == 'uint8Node':
        node.base_path = uint8Path
      elif node.label == 'maskNode':
        node.base_path = maskPath
      elif node.label == 'material':
        node.base_path = materialPath

  @abstractmethod
  def process(self, nbOfModels, fixedDistanceBtwModels, distanceBtwModels, useSameMocap):
      pass
                           
  @abstractmethod
  def process_comb(self, nbOfModels, fixedDistanceBtwModels, distanceBtwModels, useSameMocap):
      pass


#################
class RecordingPlan():
  """Class to implement different types of recording plans.

  The class was previously designed to record from top and frontal cameras.
  Currently, it only works from frontal cameras.
  """
  def __init__(self, params, dummy_mode='TOPVIEWS'):
    """Initialization of Horizontal rendering pipeline."""
    self.renderPipe = None
    # HORIZONTALVIEWS
    self.renderPipe = HorizontalRenderPipeline(params)

  def executeHorizontalMode(self, 
      cameraPosition, 
      lookatPoint, 
      cameraName, 
      airlock):
    """Execute rendering pipeline with paramters.

    Args:
      cameraPosition: Position of the virtual camera.
      lookatPoint: Orientation of the virtual camera.
      cameraName: Name of the camera.
      airlock: Airlock object to create ground floor. Note that this can be
        removed since ground is filtered using the mask of the images during 
        training.
    """
    
    # Cleans all objects from the previous rendering
    self.renderPipe.reset_blend()
    self.renderPipe.loadAndRetarget(False)
    # The walls have to be recreated after each cleaning of the file 
    # NOTE ground can be removed since it is removed during training data
    airlock.createGround(10000)
    self.renderPipe.addCamera(cameraPosition, cameraName, lookatPoint)
    self.renderPipe.extractAnnotations()

  class Params():
    def __init__(self):
      self.retargetingInfos = {}
      self.annPath = ''
      self.hdrPath = ''
      self.maskPath = ''
      self.uint8Path = ''
      self.uint16Path = ''
      self.referencePoint = None
      self.cameraPars = None



