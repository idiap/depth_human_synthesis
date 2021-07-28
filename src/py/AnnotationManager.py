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

"""Methods to dump annotations into XML files."""


from xml.dom import minidom as dom


class Person():
  """Class to contain the annotation of the Blender `armatature` (person).""" 
  class Params:
    def __init__(self):
      self.personId = -1
      self.boneData = None
      self.materialColor = None
      self.mocapFile = ''
      self.modelFile = ''

  def __init__(self, params):
    """Constructor of person class.

    Args:
      params: Person paramters.
    """
    self.id = params.personId
    # List of dictionnaries, one for each bone
    self.data = params.boneData
    self.materialColor = params.materialColor
    self.mocapFile = params.mocapFile
    self.modelFile = params.modelFile


class Frame():
  """Class to contain the list of the people's data to be saved."""
  def __init__(self, frameId, personList):
      self.id = frameId
      self.pList = personList


class AnnotationManager():
  """Class to manage annotation extraction and dumpt to XML file."""

  def __init__(self):
    """Constructor. 

    Creates the xml document, prepares the annotation subtree and append 
    the document with the root node.
    """
    self.xmlDoc = dom.Document()
    self.root = self.xmlDoc.createElement('Annotations')        
    self.animation = self.xmlDoc.createElement('animation')
    self.xmlDoc.appendChild(self.root)

  def vectorToString(self, vector):
    """Converts a vector of numbers to a string.

    Args:
      vector: Vector to be converted to string.
    """
    strVec = ''
    for p in vector:
        strVec = strVec + ' ' + str(p)            
    return strVec

  def matrixToString(self, mat):
    """Converts a matrix of numbers to a string.

    Args:
      mat: Matrix to be converted to string.
    """
    matStr = ''
    for row in mat.row:
      for col in row:
        matStr = matStr + ' ' + str(col)
    return matStr

  def setFileIds(self, filesId, modelFileName, mocapFileName):
    """Saves the full path of the .mhx2 and .bvh files in the annotation file.
    
    Args:
        modelFileName: Full path of the .mhx2 file.
        mocapFileName: Full path of the .bvh file.
    """
    idsNode = self.xmlDoc.createElement('files')
    idsNode.setAttribute('id', str(filesId))
    modelFileNode = self.xmlDoc.createElement('modelFile')
    modelFileNode.setAttribute('name', modelFileName)
    mocapFileNode = self.xmlDoc.createElement('mocapFile')
    mocapFileNode.setAttribute('name', mocapFileName)
    
    idsNode.appendChild(modelFileNode)
    idsNode.appendChild(mocapFileNode)
    self.root.appendChild(idsNode)

  def setCamera(self, camera):
    """Writes the camera parameters in the annotation file.

    The process effectively appends the document's root node with the 
    camera's node.

    Args:
      camera: Camera object containing the intrinsic and extrinsic camera parameters.
    """
    camNode = self.xmlDoc.createElement('scenecamera')

    size = self.xmlDoc.createElement('resolution')
    size.setAttribute('width', str(camera.par.widthpx))
    size.setAttribute('height', str(camera.par.heightpx))
    size.setAttribute('scale', str(camera.par.scale))
    size.setAttribute('aspectratio', str(camera.par.aspectratio))

    physical = self.xmlDoc.createElement('physical')
    physical.setAttribute('lensmm', str(camera.par.lensmm))
    physical.setAttribute('widthccdmm', str(camera.par.widthccdmm))
    physical.setAttribute('heightccdmm', str(camera.par.heightccdmm))

    pos = self.genVectorNode(camera.pos, 'worldpos', 'xyz')
    rot = self.genVectorNode(camera.generateCameraRotation(), 'roteuler', ['yaw', 'pitch', 'roll'])

    calibration = self.genMatrixNode(camera.K, 'calibrationmatrix')
    rotW = self.genMatrixNode(camera.R, 'coordrotationmatrix')
    cammatrix = self.genMatrixNode(camera.P, 'projectionmatrix')
    worldmatrix = self.genMatrixNode(camera.W, 'matrixworld')

    trans = self.genVectorNode(camera.T, 'translationvector', 'xyz')

    camNode.appendChild(pos)
    camNode.appendChild(rot)
    camNode.appendChild(size)
    camNode.appendChild(physical)
    camNode.appendChild(calibration)
    camNode.appendChild(rotW)
    camNode.appendChild(trans)
    camNode.appendChild(cammatrix)
    camNode.appendChild(worldmatrix)

    self.root.appendChild(camNode)

  def setFrame(self, dataFrame):
    """Appends the annotation node with the data of the sequence frame.

    Args:
      dataFrame: Frame containing the annotations of the armature of the
        people in the current Mocap sequence's frame.
    """
    frame = self.xmlDoc.createElement('frame')
    frame.setAttribute('id', str(dataFrame.id))
           
    for person in dataFrame.pList:
      personNode = self.xmlDoc.createElement('Person')
      personNode.setAttribute('id', str(person.id))
      personNode.setAttribute('colormaterial', str(person.materialColor))
      personNode.setAttribute('modelname', person.modelFile)
      personNode.setAttribute('mocapseq', person.mocapFile)

      bones = self.xmlDoc.createElement('bones')
      
      # List of dictionnaries, one for each bone
      for data in person.data:                   
        bone = self.xmlDoc.createElement('bone')
        bone.setAttribute('name', data['name'])

        # extract the eyes 
        if data['name'] == 'Eye1' or data['name'] == 'Eye2':
          eyeVec = self.genVectorNode(data['boneLoc'], 'worldvector', 'xyz')
          headWorldVec = self.genVectorNode(
              data['boneLoc'], 'worldheadvector', 'xyz')
          tailWorldVec = self.genVectorNode(
              data['boneLoc'], 'worldtailvector', 'xyz')
          eyeImgVec = self.genVectorNode(
              data['boneLocImg'], 'imagevector', 'xy')
          tailImgVec = self.genVectorNode(
              data['boneLocImg'], 'imgtailvector', 'xy')
          headImgVec = self.genVectorNode(
              data['boneLocImg'], 'imgheadvector', 'xy')

          bone.appendChild(eyeVec)
          bone.appendChild(headWorldVec)
          bone.appendChild(tailWorldVec)
          bone.appendChild(eyeImgVec)
          bone.appendChild(tailImgVec)
          bone.appendChild(headImgVec)
        # extract the rest of joints 
        else:
          wVector = self.genVectorNode(data['boneLoc'], 'worldvector', 'xyz')
          bMatrix = self.genMatrixNode(data['boneMatrix'], 'bonematrix')
          bImgVec = self.genVectorNode(data['boneLocImg'], 'imagevector', 'xy')
          tailVec = self.genVectorNode(
              data['boneTail'], 'worldtailvector', 'xyz')
          tailImgVec = self.genVectorNode(
              data['tailLocImg'], 'imgtailvector', 'xy')
          headVec = self.genVectorNode(
              data['boneHead'], 'worldheadvector', 'xyz')
          headImgVec = self.genVectorNode(
              data['headLocImg'], 'imgheadvector', 'xy')

          # Ajouter camera
          #camVec = self.genVectorNode(data['boneLocCam'], 'cameravector', 'xyz')
          #tailCamVec = self.genVectorNode(data['tailLocCam'], 'camtailvector', 'xyz')
          #headCamVec = self.genVectorNode(data['headLocCam'], 'camheadvector', 'xyz')

          bone.appendChild(wVector)
          bone.appendChild(bMatrix)
          # Maybe saving the world matrix is useful?
          #bone.appendChild(matrixworld)
          #bone.appendChild(camVec)
          bone.appendChild(bImgVec)
          bone.appendChild(tailVec)
          bone.appendChild(tailImgVec)
          # bone.appendChild(tailCamVec)
          bone.appendChild(headVec)
          bone.appendChild(headImgVec)
          # bone.appendChild(headCamVec)

        bones.appendChild(bone)

      # Append with all bones and the person data
      personNode.appendChild(bones)           
      frame.appendChild(personNode)

    self.animation.appendChild(frame)

  def genVectorNode(self, vector, nodeName, axisNames):
    """Generates a node based on the data of a given vector.

    The data will be appened by adding a new node for each of 
    the elements in the vector.

    Args:
      vector: Vector of data to be stored.
      nodeName : String of the node's name to be used for saving.
      axisNames: String vector of labels for the axis for the vector's elements.
    """
    v = self.xmlDoc.createElement(nodeName)
    v.setAttribute('dimension', str(len(vector)))

    for i in range(len(vector)):
      node = self.xmlDoc.createElement(axisNames[i])
      node.setAttribute('value', str(vector[i])) 
      v.appendChild(node)

    return v

  def genMatrixNode(self, mat, nodeName):
    """Generates a new node based on the information of the matrix.

    Args:
      mat: Matrix of data to be stored.
      nodeName: String of the node's name to be used for saving
    """
    m = self.xmlDoc.createElement(nodeName)
    m.setAttribute('rows', str(len(mat.row)))
    m.setAttribute('cols', str(len(mat.col)))

    i = 0
    for row in mat.row:
      rowStr = self.vectorToString(row)
      node = self.xmlDoc.createElement('row_'+str(i))
      node.setAttribute('value', rowStr)
      m.appendChild(node)
      i = i + 1

    return m

  def writeToFile(self, xmlFileName):
    """Writes the extracted annotations to a XML file.

    Args:
      xmlFileName: Full path name of the output file.
    """
    # First, append the root with the annimation annotation
    # TODO do the append somewhere else
    self.root.appendChild(self.animation)
    xmlStr = self.xmlDoc.toprettyxml(indent='    ')
    with open(xmlFileName, "w") as f:
      f.write(xmlStr)
