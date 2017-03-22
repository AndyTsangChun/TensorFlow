# A Set-up class for getting ATT dataset
# Implemented in Python 2.7
# By Andy Tsang

import numpy as np
import pickle
import os
import re
import sys
import helper.download as dl
from PIL import Image

# Static Variables
DATA_PATH = 'data/ATT/'
DATA_ATT_PATH = 'orl_faces/'
DATA_URL = 'http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.tar.Z'
# Constants
# Original size is 92 x 112, however to convenient CNN use 100 x 100 instead
IMAGE_SIZE = 100
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
NUM_CHANNELS = 1
IMAGE_SIZE_FLAT = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS
NUM_CLASSES = 40
# Variables about AT&T dataset
FOLDER_PREFIX = 's'
NUM_FOLDER_IMAGE = 10
TRAINING_SIZE = 8
TEST_SIZE = NUM_FOLDER_IMAGE - TRAINING_SIZE

#====================================================================
# Helper Function

# Helper-Function to create summaries for activations.
#Input:
#   x: Tensor
#Return:
#   None
def activation_summary(x):
  tf.summary.histogram(x.op.name + '/activations', x)
  tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

# Helper-Function to convert class to one-hot representation
#Input:
#   class_number: Classes of images
#   num_classes: Total number of classes
#Return:
#   one-shot representation of classes
def one_hot_encoded(class_numbers, num_classes=None):
  if num_classes is None:
    num_classes = np.max(class_numbers) - 1
    
  return np.eye(num_classes, dtype=float)[class_numbers]

#====================================================================
# Private Function

# Get data from traget file
#Input:
#   filename: String
#Return:
#   image: [height, width, channel]
def _get_data_array(filename):
  file_path = os.path.join(DATA_PATH, DATA_ATT_PATH, filename)
  with Image.open(file_path) as file:
    # Opening the image and convert into numpy array
    image = _reshape_image(file)
    image = np.array(image)
    image = _add_channel(image)

  return image

# Reshape image from 92 x 112 to 100 x 100
#Input:
#   image: 91 x 112
#Return:
#   image: 100 x 100 [height, width]
def _reshape_image(image):
  return image.resize(IMAGE_SHAPE)

# For grey scale image, it has only 1 channel and doesn't comes along
# with the array. Therefore, we had to add the channel by ourself.
#Input:
#   image: [height, width]
#Return:
#   image: [height, width, channel]
def _add_channel(image):
  image = np.expand_dims(image, axis=2)
  return image

# Return the number of image used base on is training or not
#Input:
#   isTraining: boolean
#Return:
#   images_no: number reference of image
def _get_image_no(isTraining):
  if (isTraining):
    return TRAINING_SIZE
  else:
    return TEST_SIZE

# Load data from target file and return images along with its class
#Input:
#   foldernum: int
#   isTraining: boolean
#Return:
#   images: [image_number, height, width, channel]
#   cls: class of the images
def _load_class_data(foldernum, isTraining):
  image_no = _get_image_no(isTraining)
  # Get data from file.
  folder_path = FOLDER_PREFIX + foldernum + '/'
  # Initialise empty array for images
  images = np.zeros(shape=[image_no,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS], dtype=float)
  # Set all image from that folder to the same classes
  cls = np.ones(shape=[image_no,], dtype=int) * (int(foldernum) - 1)
  
  # Base on its training or not, determine the index number of image
  # AT&T dataset contain 10 images for each class
  # Assume Training-set start from file 1
  # While Test-set start from file 1 + training-set size
  if(isTraining):
    index = 1
  else:
    index = 1 + TRAINING_SIZE
  
  # Get AT&T face and convert into array
  for i in range(image_no):
      images[i,:,:,:] = _get_data_array(folder_path + str(index + i) + '.pgm')

  return images, cls

#====================================================================
# Public Functions

# Using function from download to acquire and extract the dataset
#Input:
#   None
#Return:
#   None
def maybe_download_and_extract():
  # Check dataset exist or not
  if not os.path.exists(DATA_PATH + DATA_ATT_PATH):   
    dl.maybe_download_and_extract(data_url=DATA_URL, data_path=DATA_PATH)
  else:
    print("Data has already been extracted in previous session.")

# Since the AT&T dataset don't have specific training or test set. We have
# to split it by ourself.
#Input:
#   isTraining: boolean
#Return:
#   images: Collection of images [image_number, height, width, channel]
#   cls: Collection of class 
#   cls_one_hot: Collection of class in one-hot representation
def get_data(isTraining):
  image_no = _get_image_no(isTraining)
  # Initialise array for images and classes
  images = np.zeros(shape=[NUM_CLASSES * image_no, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], dtype=float)
  cls = np.zeros(shape=[NUM_CLASSES * image_no], dtype=int)
  cls_one_hot = np.zeros(shape=[NUM_CLASSES * image_no, NUM_CLASSES], dtype=int)

  begin = 0
  for i in range(NUM_CLASSES):
    # Getting the image from each class folder and group them together
    sub_image, sub_cls = _load_class_data(foldernum = str(i + 1), isTraining = isTraining)
    # Getting the end number
    end = begin + image_no
    # Add images and class subset to the summarised array
    images[begin:end, :] = sub_image
    cls[begin:end] = sub_cls
    # Update the next begin number
    begin = end
  # Encode class into one-hot
  cls_one_hot = one_hot_encoded(class_numbers=cls, num_classes=NUM_CLASSES)
  print("Training data acquire successfully!")
  print("The Training-set contain [" + str(len(images)) + "] images")

  return images, cls, cls_one_hot

# Use for testing, get specific image from specific class
#Input:
#   classno: Class number
#   imageno: Image number
#Output:
#   image: [height, width, channel]
def get_one_image(classno, imageno):
  filename = 's'+ str(classno) + '/' + str(imageno) + '.pgm'
  image = _get_data_array(filename)

  return image

