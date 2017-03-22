import tensorflow as tf

# Help-Function that pre-processes input images.
# Increase number of image in training-set and test-set by adding distortion.
# The process is different for training-set and test-set
#
# For Training-set, input images are randomly cropped, randomly flipped horizontally,
# Since we only have 1-channel other function such as hue, contrast and saturation 
# is disabled. This artificially inflates the size of the training-set by creating
# random variations of the original input images. 
#
# For Test-set, input images are only cropped around centre cause its not necessary to
# add other color related variations.
#
# Input:
#   image: Original images
#   isTraining: boolean determine whether the input images is training-set
#               or test-set
# Output:
#   image: Distorted images
IMG_SIZE_CROPPED = 80

def _image_distort(image, isTraining):
    num_channels = int(image.shape[2])
    if isTraining:
        # For training, add the following to the TensorFlow graph.
        
        # Randomly crop the input image.
        image = tf.random_crop(image, size=[IMG_SIZE_CROPPED, IMG_SIZE_CROPPED, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        #image = tf.image.random_hue(image, max_delta=0.05)
        #image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        #image = tf.image.random_brightness(image, max_delta=0.2)
        #image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        #image = tf.minimum(image, 1.0)
        #image = tf.maximum(image, 0.0)
    else:
        # For test, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image, target_height=img_size_cropped, target_width=img_size_cropped)

    return image 

def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: _image_distort(image, training), images)

    return images
    