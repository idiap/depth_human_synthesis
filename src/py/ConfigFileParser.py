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

"""Read configuration file provided in XML file format."""

import sys
import os
import Utils
import numpy as np
import xml.etree.ElementTree as ET

# Configuration file parser.
class ConfigFileParser():
  """This class contains the methods to read the configuration file and to
  get the configuration parameters. The configuration file is a xml file."""

  class ImportExportParams():
    """This class contains the values of the configuration parameters."""
    # TODO (amartinez) change this class for dict objects
    def __init__(self):
      """Init parameters to empty values."""
      self.modelPath = ''  # Path to the model files.
      self.mocapSeqPath = ''  # Path to the mocap sequences.
      self.outputPath = ''  # Output dataset path.
      self.mocapFileListing = ''  # Pre-selected list of mocap sequences.
      self.modelFileList = []  # List of model files.
      self.mocapFileList = []  # List of mocap files.
      self.statusRecovery = '' # File for recovery from previous run

  def __init__(self, xmlFilePath):
    """Sets the path to the XML configuration file and does sanity checks.
    Args:
      xmlFilePath: Full path to the main configuration file.
    """
    # Check if the input xml file exists.
    Utils.fun_check_file(xmlFilePath)
    # Set configuration file path.
    self.xmlFilePath = xmlFilePath
    # Message.
    Utils.fun_messages(
        'Main configuration file path: {}'.format(self.xmlFilePath), 'info')

  # Get configuration parameters.
  def getParams(self, sort=False):
    """Reads the configuration file and returns the parameters values.

    Args:
      sort: Bool variable to enable sorting model and mocap file lists.

    Returns:
      ImportExportParams instance with configuration paramters.
    """
    # Get xml tree.
    tree = ET.parse(self.xmlFilePath).getroot()

    # Create parameters class -empty values-.
    params = self.ImportExportParams()

    # Get parameters.
    params.modelPath = tree.find('inputs').\
            find('humanmodelspath').get('path')  # Human model files path.
    params.mocapSeqPath = tree.find('inputs').\
            find('mocapdatasetpath').get('path')  # Mocap files path.
    params.outputPath = tree.find('outputs').\
            find('destinationpath').get('path')  # Output path.
    params.mocapFileListing = tree.find('inputs').\
            find('mocapselectedlist').get('path') # Pre-select mocap list.
    params.statusRecovery = tree.find('inputs').find('statusRecoveryFile').get('path')

    # Check the configuration parameters.
    Utils.fun_check_directory(params.modelPath)
    Utils.fun_check_directory(params.mocapSeqPath)
    Utils.fun_check_file(params.mocapFileListing)

    # Get a list of model files (mhx2 files).
    params.modelFileList = ConfigFileParser.getFileList(params.modelPath, \
                                                        'mhx2')
    # Get a list of mocap files (bhv files) according to the pre-selected
    # mocap files list.
    if params.mocapFileListing is '':
      params.mocapFileList = \
          ConfigFileParser.getFileList(params.mocapSeqPath, 'bvh')
    else:
      params.mocapFileList = \
          ConfigFileParser.getFileListFromFile(params.mocapFileListing)

    # Sort model and mocap lists.
    if sort:
      params.modelFileList.sort()
      params.mocapFileList.sort()

    # Print parameters.
    Utils.fun_messages('Human models path: {}'.format(params.modelPath), 'info')
    Utils.fun_messages(
        'Mocap sequences path: {}'.format(params.mocapSeqPath), 'info')
    Utils.fun_messages(
        'Mocap selected list: {}'.format(params.mocapFileListing), 'info')
    Utils.fun_messages('Output path: {}'.format(params.outputPath), 'info')

    return params

  @staticmethod
  def getFileList(folderPath, fileExtension):
    """Gets the files in a folder and returns them in a list of paths.

    Args:
      folderPath: Full path of the folder to read in.
      fileExtension: Extension of the files.

    Returns:
      A list of path files contained in folderPath
    """

    # Check paths.
    Utils.fun_check_directory(folderPath)

    # Compute file list.
    file_list = [f for f in os.listdir(folderPath) 
        if os.path.isfile(os.path.join(folderPath, f)) and f.endswith(fileExtension)]

    # Check list.
    if len(file_list)==0: Utils.fun_messages('Empty list', 'error')

    return file_list

  @staticmethod
  def getFileListFromFile(fileListing, shuffle=True):
    """Reads a file listing the mocap/model files to use

    The full path has to be provided and each line contains different files.

    Args:
        fileListing: File containing the selected files.
        shuffle: Enable shuffling file list.

    Returns:
      A list of files.
    """
    # Check file.
    Utils.fun_check_file(fileListing)
    Utils.fun_messages('Loading files from listing file', 'process')

    file_list = []

    with open(fileListing) as file_:
      for line in file_:
        file_list.append(line[:-1])

    # Check list.
    if len(file_list)==0: Utils.fun_messages('Empty list', 'error')

    # Shuffle file list.
    if shuffle: np.random.shuffle(file_list)

    return file_list
