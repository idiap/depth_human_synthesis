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

"""Methods to parse and configure camera parameters and rendering traits in blender."""


import sys
sys.path.append('.')
import Utils
from Utils import fun_messages
import xml.etree.ElementTree as ET


class RecManagerParser():
  """This class contains the classes and methods to read the XML configuration
  file and the configuration parameters such as camera traits, and
  reference point parameters.
  """

  class CameraParams():
    """Camera traits parameters."""
    def __init__(self):
      """Sets default values as set in Blender GUI."""
      #TODO Add more related camera features
      self.name = 'default'
      # Resolution and focal length -default blender values-.
      self.lensmm = 35.0
      self.widthpx = 1920
      self.heightpx = 1080
      # Physical sensor (CCD) size -default values-.
      self.widthccdmm = 32
      self.heightccdmm = 18
      self.aspectratio = 1.0
      self.scale = 0.5
      self.resratio = 50.0
      self.sensorfit = 'AUTO'

  #TODO (amartinez): This class could be just a vector
  class referencePointParams():
    """This class contains the reference point values."""
    def __init__(self):
      """Initializes the reference point parameters to default values."""
      # Reference point.
      self.x = 0.0
      self.y = 0.0
      self.z = 0.0

  class AirlockParams():
    """This class contains the parameters related to the airlock."""
    def __init__(self):
      """This function initializes the airlock parameters to default values."""
      # Airlock dimension.
      self.dimX = 0.0
      self.dimY = 0.0
      # Minimum/maximum airlock height.
      self.minDimZ = 0.0
      self.maxDimZ = 0.0
      # Step between each camera.
      self.stepX = 0.0
      self.stepY = 0.0
      # Wall values.
      self.wallA_transp = 0.0
      self.wallB_transp = 0.0
      self.wallC_transp = 0.0
      self.wallD_transp = 0.0

  def __init__(self, xmlFilePath):
    """Sets path to confituration file.

    Args:
        xmlFilePath: Path to the recording manager configuration file.
    """
    # Check if the input xml file exists.
    Utils.fun_check_file(xmlFilePath)
    # Set configuration file path.
    self.xmlFilePath = xmlFilePath

    # Message.
    fun_messages(
        'Recording configuration file path: {}'.format(self.xmlFilePath), 'info')

  def getParams(self):
    """Returns configuration parameters of camera, airlock and reference point."""

    # Parse the configuration file.
    self.parseConfigFile()

    # All configuration parameters.
    params = (self.referencePointParams, self.airlockParams, self.camParamsList)

    return params

  # Parse the configuration file.
  def parseConfigFile(self):
    """Parses the configuration file."""

    # Get xml tree root.
    root = ET.parse(self.xmlFilePath).getroot()

    # Get reference point parameters.
    self.referencePointParams = \
        self.parseReferencePoint(root.find('referencepoint'))

    # Get airlock parameters.
    self.airlockParams = self.parseAirlockTree(root.find('airlock'))

    # Get camera parameters. Get the parameters for all cameras defined in the 
    # configuration file (xml file)
    self.camParamsList = []
    for cameraNode in root.find('cameras'):
      self.camParamsList.append(self.parseCameraTree(cameraNode))

  # Reference point parser.
  def parseReferencePoint(self, refPointNode):
    """Parses the reference point parameters.

    Args:
      refPointNode: Tree node to the reference point parameters in xml file.

    Returns:
      Reference point paramters.
    """

    # Get current reference point parameters.
    params = self.referencePointParams()

    # Get parameters defined in the configuration file.
    params.x = float(refPointNode.get('x'))
    params.y = float(refPointNode.get('y'))
    params.z = float(refPointNode.get('z'))

    # Print parameters.
    fun_messages('Reference point x: {}'.format(params.x), 'info')
    fun_messages('Reference point y: {}'.format(params.y), 'info')
    fun_messages('Reference point z: {}'.format(params.z), 'info')

    return params

  def parseAirlockTree(self, airlockNode):
    """Parses the airlock parameters.

    Args:
      airlockNode: Tree node to the airlock parameters in xml file.

    Returns:
      Airlock paramters.
    """

    # Get current airlock parameters.
    params = self.AirlockParams()

    # Get parameters defined in the configuration file.
    # Airlock dimension.
    params.dimX = float(airlockNode.find('dimx').get('value'))
    params.dimY = float(airlockNode.find('dimy').get('value'))
    params.minDimZ = float(airlockNode.find('dimz').get('minValue'))
    params.maxDimZ = float(airlockNode.find('dimz').get('maxValue'))
    # Airlock grid step.
    params.stepX = float(airlockNode.find('stepx').get('value'))
    params.stepY = float(airlockNode.find('stepy').get('value'))
    # Airlock wall.
    params.wallA_transp = float(airlockNode.find('materials').\
                                find('walla').get('transparency'))
    params.wallB_transp = float(airlockNode.find('materials').\
                                find('wallb').get('transparency'))
    params.wallC_transp = float(airlockNode.find('materials').\
                                find('wallc').get('transparency'))
    params.wallD_transp = float(airlockNode.find('materials').\
                                find('walld').get('transparency'))

    # Print parameters.
    fun_messages('Airlock dim. x: {}'.format(params.dimX), 'info')
    fun_messages('Airlock dim. y: {}'.format(params.dimY), 'info')
    fun_messages(
        'Airlock dim. z: [{} - {}]'.format(params.minDimZ, params.maxDimZ), 'info')
    fun_messages('Airlock step x: {}'.format(params.stepX), 'info')
    fun_messages('Airlock step y: {}'.format(params.stepY), 'info')
    fun_messages(
        'Airlock wall a transparency: {}'.format(params.wallA_transp), 'info')
    fun_messages(
        'Airlock wall b transparency: {}'.format(params.wallB_transp), 'info')
    fun_messages(
        'Airlock wall c transparency: {}'.format(params.wallC_transp), 'info')
    fun_messages(
        'Airlock wall d transparency: {}'.format(params.wallD_transp), 'info')

    return params

  # Camera parser.
  def parseCameraTree(self, cameraNode):
    """Parses the camera parameters.

    Args:
      cameraNode: Tree node to the camera parametes in xml file.

    Returns:
      Camera parameters.
    """

    # Get current camera parameters.
    params = self.CameraParams()

    # Get parameters defined in the configuration file.
    params.name = cameraNode.get('name')
    params.lensmm = float(cameraNode.find('lensmm').get('value'))
    params.widthpx = int(cameraNode.find('widthpx').get('value'))
    params.heightpx = int(cameraNode.find('heightpx').get('value'))
    params.widthccdmm = int(cameraNode.find('widthccdmm').get('value'))
    params.heightccdmm = int(cameraNode.find('heightccdmm').get('value'))
    params.sensorfit = cameraNode.find('sensorfit').get('value')
    params.resratio = float(cameraNode.find('resolutionpercentage').\
                            get('value'))

    # Print parameters.
    fun_messages('camera name: {}'.format(params.name), 'info')
    fun_messages('camera lens [mm]: {}'.format(params.lensmm), 'info')
    fun_messages('camera width [pixels]: {}'.format(params.widthpx), 'info')
    fun_messages('camera height [pixels]: {}'.format(params.heightpx), 'info')
    fun_messages('camera ccd width [mm]: {}'.format(params.widthccdmm), 'info')
    fun_messages('camera ccd height [mm]: {}'.format(params.heightccdmm), 'info')
    fun_messages('camera sensor fit: {}'.format(params.sensorfit), 'info')
    fun_messages('camera resolution ratio: {}'.format(params.resratio), 'info')

    return params
