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

"""Implements the recording and rendering of depth images with annotations.

Data is process in pairs of (3D characters, mocap sequences) by first creating
a product between all available 3D characters and all mocap sequences. Then each
pair is processed for rendering following the rendering pipeline. Images are 
either saved in EXR format (float32) or tiff format (uint16).
"""

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

from BaseRecordingPlan import BaseRecordingManager
from BaseRecordingPlan import RecordingPlan
from HorizontalRenderPipeline import HorizontalRenderPipeline as DepthRenderPipeline
from CameraModule import Camera
from RecordingZone import RecordingZone 
import Utils

_NCAMERAS = 3
_HIPS_HEIGHT = 1.5
_NECK_HEIGHT = 1.8


class HorizontalRecordingManager(BaseRecordingManager):
  """Implements the synthesize of depth images from a sensor in front of poeple.

  This class implements the recording plan to synthesize depth images from
  a up-right view, simulating a depth sensor in front of people such as in
  human robot interactions.
  """
  def __init__(self, configParams, recParams, nbOfModels):
    """Constructor and initalization of base class."""
    BaseRecordingManager.__init__(self, configParams, recParams, nbOfModels)

  def recover_from_prev_run(self):
    """Recover status from previous run logged into a txt file or from scratch.

    Previous stopped execution of the rendering pipeline could be recovered from
    the log file (a txt file). This method reads the log and generates the 
    list of (mocap_sequence, 3d_character) pairs that should follow the run.
    """
    modelMocapPairs = []
    print('[INFO] The status recovery file is', self.configParams.statusRecovery)

    if self.configParams.statusRecovery == '' or self.configParams.StatusRecovery == None:
      # if no previous run log file is provided, build initial list
      # For each model file build all combinations of models and mocap files.
      # Re-shuffling to have variety and not depend on the order of model listing.
      modelMocapPairs = list(
          product(self.configParams.modelFileList, 
                  self.configParams.mocapFileList)
      )
      npy.random.shuffle(modelMocapPairs)
    else:
      with open(self.configParams.statusRecovery) as file_:
        for line in file_:
          l = len(line)
          line = (line[0:l-1]).split(' ')
          modelMocapPairs.append((line[0], line[1]))

      print('[INFO] Resuming from previous recovery from file', 
          self.configParams.statusRecovery)
      print('[INFO] Starting from pair', modelMocapPairs[1])
      modelMocapPairs = modelMocapPairs[1:]

    return modelMocapPairs

  def select_second_person_model_and_mocap(self, modelFile1, mocapFile1):
    """Randomly selects an model and mocap file for second model.

    Args:
      modelFile1: Path for the first 3D human character.
      mocapFile1: Path for the first mocap sequence file.

    Returns:
      3D character file path and mocap sequence path.
    """
    modelFile2 = random.choice(self.configParams.modelFileList)
    while modelFile2 == modelFile1:
      modelFile2 = random.choice(self.configParams.modelFileList)

    mocapFile2 = random.choice(self.configParams.mocapFileList)

    return modelFile2, mocapFile2

  def define_second_model_position(self, 
      position, 
      fixedDistanceBtwModels, 
      distanceBtwModels):
    """Selects second model position such that models don't collide.

    Args:
      position: Position of first model.
      fixedDistanceBtwModels: Boolean indicated if fixed distance between the
        3D characters should be held during the recording.
      distanceBtwModels: Distance to keep between models.

    Returns:
      Vector of position for the second model.
    """
    position2 = position
    if fixedDistanceBtwModels:
        angle = npy.random.vonmises(0, 0, size=None) 
        position2 = position + mathutils.Vector(
            [distanceBtwModels * math.cos(angle),
             distanceBtwModels * math.sin(angle),
             0]
        ) 
    else:          
        position2 = position + mathutils.Vector(
            [npy.random.uniform(low=0.4, high=distanceBtwModels),
             npy.random.uniform(low=0.4, high=distanceBtwModels),
             0]
        )

    return position2

  def create_output_paths(self, outputPath, mocapFilePath, cameraName):
    """Create output tree to save models and annotations."""
    self.uint8Path = os.path.join(outputPath, 'depth_uint8_imgs')
    self.hdrPath = os.path.join(outputPath, 'depth_hdr_imgs')
    self.uint16Path = os.path.join(outputPath, 'depth_uint16_imgs')
    self.maskPath = os.path.join(outputPath, 'obj_mask_imgs')
    self.annPath = os.path.join(outputPath, 'annotations')
    self.materialPath = os.path.join(outputPath, 'obj_material_imgs')
   
    # create tree 
    os.makedirs(self.uint8Path, exist_ok=True)
    os.makedirs(self.hdrPath, exist_ok=True)
    os.makedirs(self.annPath, exist_ok=True)
    os.makedirs(self.maskPath, exist_ok=True)
    os.makedirs(self.uint16Path, exist_ok=True)
    os.makedirs(self.materialPath, exist_ok=True)

  def save_processing_status(self, modelMocapPairs, ii):
    """Dumps in a log current remainder pairs of 3D characters and Mocap pairs."""
    # Write reminder of pairs to be processed
    reminders = modelMocapPairs[(ii+1):len(modelMocapPairs)]
    with open('./StatusRecovery.txt', 'w') as file_:
      for pair in reminders:
        line = '%s %s' % (pair[0], pair[1])
        print(line, file=file_)

  def process_comb(self,
                   minDistanceToCamera,
                   nbOfModels,
                   fixedDistanceBtwModels,
                   distanceBtwModels,
                   useSameMocap,
                   convertToUint16=False):
    """Executes rendering pipeline and saves images and annotations.

    This is the main function to perform data synthesis. The method first
    generate a list of pairs (3D character, Mocap sequence) and process each
    pair at a time following the rendering pipeline, i.e. number of cameras,
    random rotations, etc, to finally extract annotations.

    Args:
      minDistanceToCamera: Minimum distance between models and the camera.
      nbOfModels: Number of models to consider in the rendering pipeline.
      fixedDistanceBtwModels: Boolean indicating if a fixed distance between
          models should be kept during rendering.
      useSameMocap: Boolean indicating if the two models should use the same
          Mocap sequence.
      convertToUint16: Boolean indicating if convertion to uint16 image format
          should be performed.
    """
    self.prepareRenderNodes()
    # Construct reference point
    referencePoint = mathutils.Vector(
        [self.referencePointParams.x,
         self.referencePointParams.y, 
         self.referencePointParams.z]
    )
    # Constructs the scene to be rendered
    airlock = RecordingZone(referencePoint, self.airlockParams)

    # Recover from previous run?
    modelMocapPairs = self.recover_from_prev_run()

    # Loop over all paired mocap and model files
    for ii in range(len(modelMocapPairs)):
      pair = modelMocapPairs[ii]
      modelFile, mocapFile = pair[0], pair[1]

      modelFileName = os.path.splitext(modelFile)[0]
      modelFilePath = os.path.join(self.configParams.outputPath, modelFileName)

      #if not os.path.exists(modelFilePath):
      os.makedirs(modelFilePath, exist_ok=True)

      # Select a second model file in case of rendering with more than 1 person
      if nbOfModels == 2:
        modelFile2, mocapFile2 = self.select_second_person_model_and_mocap(
            modelFile, mocapFile
        )

      mocapFileName = os.path.basename(mocapFile).split('.')[0]
      mocapFilePath = os.path.join(modelFilePath, mocapFileName)

      #if not os.path.exists(mocapFilePath):
      os.makedirs(mocapFilePath, exist_ok=True)

      # For each camera configuration paramters create at maximum 3 random
      # cameras around the models to perform rendering and annotation extraction.
      for pars in self.camParamsList:                                                         
        recordingParams = RecordingPlan.Params()                             
        recordingParams.cameraPars = pars  
        recordingParams.referencePoint = referencePoint   

        # TODO ADD RANDOMNESS to z position of reference point? 
        position = mathutils.Vector([
            self.referencePointParams.x + npy.random.uniform(low=0.0, high=0.3), 
            self.referencePointParams.y + npy.random.uniform(low=0.0, high=0.3), 
            self.referencePointParams.z]
        )
                                     
        # (modelFile, mocapFile, modelPosition)   
        key = (os.path.splitext(modelFile)[0]).lower().capitalize()                                            
        recordingParams.retargetingInfos[key] = (
            os.path.join(self.configParams.modelPath, modelFile),
            os.path.join(self.configParams.mocapSeqPath, mocapFile),
            position
        )
                            
        if nbOfModels == 2: 
          position2 = self.define_second_model_position(
              position, fixedDistanceBtwModels, distanceBtwModels)
          key = (os.path.splitext(modelFile2)[0]).lower().capitalize()
          recordingParams.retargetingInfos[key] = (
              os.path.join(self.configParams.modelPath, modelFile2), 
              os.path.join(self.configParams.mocapSeqPath, mocapFile2),
              position2
          )
          airlock.referencePoint = (position + position2) /2.0

        # Generate random positions of cameras around the model
        # Distance to referencePoint, minHeight, maxHeight, nbOfPositions
        hipsHeight, neckHeight = _HIPS_HEIGHT, _NECK_HEIGHT
        positions = airlock.createCameraHorizontalPosition(
            hipsHeight, neckHeight, _NCAMERAS)

        i = 1
        for cameraPosition, lookatPoint in positions:
          cameraName = pars.name + '_' + str(i)
          #lookatPoint = position # This will fix the camera to a given height
          print('[INFO] The camera position is', cameraPosition)
          print('[INFO] The lookatPoint is', lookatPoint)
                                
          outputPath = os.path.join(mocapFilePath, cameraName)
          self.create_output_paths(outputPath, mocapFilePath, cameraName)
                                                  
          self.setNodePaths(self.hdrPath, self.uint8Path, self.maskPath, self.materialPath)
          recordingParams.annPath = self.annPath
          recordingParams.hdrPath = self.hdrPath
          recordingParams.maskPath = self.maskPath
          recordingParams.uint8Path = self.uint8Path
          recordingParams.uint16Path = self.uint16Path

          # Execute recording plan
          recPlan = RecordingPlan(recordingParams, 'HORIZONTALVIEWS')
          recPlan.executeHorizontalMode(
              mathutils.Vector(cameraPosition),
              mathutils.Vector(lookatPoint),
              cameraName, airlock
          )

          i += 1
          # Convert heavy EXR float image format to unsigned int 16 bits
          if convertToUint16:
            self.convert_exr_to_uint16(self.hdrPath, self.uint16Path)

      # Write to file the remainder of paired models and mocaps to process
      self.save_processing_status(modelMocapPairs, ii)
      break

  def convert_exr_to_uint16(self, exrPath, uint16Path):
    """Converts heavy EXR float32 image to uint16 deleting previous data."""
    print('[INFO] Converting float image to uint16 image')
    print('[INFO] Processing directory', exrPath)
    print('[INFO] Saving directory', uint16Path)

    commandString = 'python ConvertTouint16.py --input_path %s --output_path %s'\
         % (exrPath, uint16Path)

    rVal = os.system(commandString)
    assert rVal == 0, "[ERROR] Convertion to uint16 script failed {}".format(rVal)

    print('[WARNING] Deleting folder', exrPath)
    shutil.rmtree(exrPath)

