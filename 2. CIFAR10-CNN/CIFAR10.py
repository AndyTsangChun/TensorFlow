# A Set-up class for getting CIFAR-10 dataset
# Implemented in Python 2.7
# By Andy Tsang

import numpy as np
import pickle
import os
import re
import sys
import download as dl

# Static Variables
DATA_PATH = 'data/CIFAR-10/'
DATA_CIFAR_10_PATH = 'cifar-10-batches-py/'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# Constants
IMAGE_SIZE = 32
NUM_CHANNELS = 3
IMAGE_SIZE_FLAT = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS
NUM_CLASSES = 10
# Variables about CIFAR-10 dataset
NUM_SEP_FILES = 5
NUM_SEP_IMAGE = 10000

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
#   data: raw data 
def _get_data(filename):
  file_path = os.path.join(DATA_PATH, DATA_CIFAR_10_PATH, filename)
  with open(file_path, mode='rb') as file:
    # If using Python 3.X it is necessary to set the encoding,
    # otherwise an exception is raised here.
    data = pickle.load(file)

  return data

# Convert the raw images from the data-files to floating-points.
#Input:
#   raw: CIFAR-10 format
#Return:
#   images: 4-dim float array [image_number, height, width, channel], range 0.0 ~ 1.0
def _convert_images(raw):
  # Divide by 255 to normalise the data
  raw_float = np.array(raw, dtype=float) / 255.0
  # Reshape the array to 4-dimensions.
  images = raw_float.reshape([-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE])
  # Reorder the indices of the array, num of image unchanged
  images = images.transpose([0, 2, 3, 1])

  return images

# Load data from target file and return images along with its class
#Input:
#   filename: String
#Return:
#   images: [image_number, height, width, channel]
#   cls: class of the images
def _load_data(filename):
  # Get data from file.
  data = _get_data(filename)
  # Get the raw images.
  raw_images = data[b'data']
  # Get the class-numbers for each image. Convert to numpy-array.
  cls = np.array(data[b'labels'])
  # Convert the images.
  images = _convert_images(raw_images)

  return images, cls

#====================================================================
# Public Functions

# Using function from download to acquire and extract the dataset
#Input:
#   None
#Return:
#   None
def maybe_download_and_extract():
  dl.maybe_download_and_extract(data_url=DATA_URL, data_path=DATA_PATH)

# Returning All Class Name
#Input:
#   None
#Return:
#   names: Name of respective class
def get_class_names():
  # Load the class-names from the pickled file.
  raw = _get_data(filename="batches.meta")[b'label_names']
  # Convert from binary strings.
  names = [x.decode('utf-8') for x in raw]
    
  return names

# Load and merge the seperated training set, returns image, class and one-hot encoded labels.
#Input:
#   None
#Return:
#   images: Collection of images [image_number, height, width, channel]
#   cls: Collection of class 
#   cls_one_hot: Collection of class in one-hot representation
def get_training_data():
  # Initialise array for images and classes
  images = np.zeros(shape=[NUM_SEP_FILES * NUM_SEP_IMAGE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], dtype=float)
  cls = np.zeros(shape=[NUM_SEP_FILES * NUM_SEP_IMAGE], dtype=int)
  cls_one_hot = np.zeros(shape=[NUM_SEP_FILES * NUM_SEP_IMAGE, NUM_CLASSES], dtype=int)
    
  begin = 0
  for i in range(NUM_SEP_FILES):
    # Getting the subset of the seperated dataset
    sub_image, sub_cls = _load_data(filename="data_batch_" + str(i + 1))
    # Getting the start number
    num_images = len(sub_image)
    # Getting the end number
    end = begin + num_images
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

# Load test set, returns image, class and one-hot encoded labels.
#Input:
#   None
#Return:
#   images: Collection of images [image_number, height, width, channel]
#   cls: Collection of class 
#   cls_one_hot: Collection of class in one-hot representation
def get_test_data():
  images, cls = _load_data(filename="test_batch")
  # Encode class into one-hot
  cls_one_hot = one_hot_encoded(class_numbers=cls, num_classes=NUM_CLASSES)
  print("Test data acquire successfully!")
  print("The Test-set contain [" + str(len(images)) + "] images")
    
  return images, cls, cls_one_hot
