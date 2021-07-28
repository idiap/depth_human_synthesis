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

"""Main script for running the synthesis pipeline for horizontal views.

The version of this code works with XML file style to read configuration files,
to setup blender and save annotations. Hyperparameters for scene, such as
number of models, distance to camera, etc. are provided by a JSON file.
"""


import sys
import json

# This is blender python
import bpy
# Our managers for recording
from ConfigFileParser import ConfigFileParser
from RecManagerParser import RecManagerParser
from HorizontalRecordingPlan import HorizontalRecordingManager as RecordingManager


def startRecording():
  """Setups recording pipeline and exectutes recording process."""
    # Activate MakeHuman addons
  bpy.ops.wm.addon_enable(module="import_runtime_mhx2")
  bpy.ops.wm.addon_enable(module="makeclothes")
  bpy.ops.wm.addon_enable(module="maketarget")
  bpy.ops.wm.addon_enable(module="makewalk")

  # This file is expected to be in the working tree
  runParamters = '../../ConfigFiles/scene_parameters.json'
  sceneParameters = json.load(open(runParamters))

  # These configurations are mostly to setup blender parameters
  mainConfigFile = sceneParameters['main_config_file']
  recManagerConfigFile = sceneParameters['rec_manager_config_file']

  minDistanceToCamera = sceneParameters['min_distance_to_camera'] # 0.7
  nbOfModels = sceneParameters['number_of_models'] #2
  fixedDistanceBtwModels = sceneParameters['fixed_models_distance'] # False
  distanceBtwModels = sceneParameters['distance_btw_models'] # 1.0
  useSameMocap = sceneParameters['use_same_mocap'] # False
  convertToUint16 = sceneParameters['convert_to_uint16']

  if nbOfModels > 2 and nbOfModels<1:
    raise ValueError("Number of models should be 1 or 2, provided: {}".\
        format(nbOfModels))

  # create parser for blender config
  configParser = ConfigFileParser(mainConfigFile)
  recManagerParser = RecManagerParser(recManagerConfigFile)
  recManager = RecordingManager(
      configParser.getParams(), 
      recManagerParser.getParams(), 
      nbOfModels
  )
  
  # If the blender application is not open, then run in background                              
  if not bpy.app.background:                              
    area = bpy.context.window_manager.windows[0].screen.areas[4]
    screen = bpy.context.window_manager.windows[0].screen
    window = bpy.context.window_manager.windows[0]
    override = {'window': window, 'screen': screen, 'area': area}
    bpy.ops.screen.screen_full_area(override)
  
  # Run synthesis process
  recManager.process_comb(
      minDistanceToCamera,
      nbOfModels,
      fixedDistanceBtwModels,
      distanceBtwModels,
      useSameMocap,
      convertToUint16=convertToUint16
  )


if __name__ == '__main__':
  startRecording()
