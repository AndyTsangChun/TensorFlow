import tensorflow as tf
import cnn_layer_util as cnn_util

#Function creates a new convolutional layer in the computational graph for TensorFlow. 
#Input:
#   input: From previous layer
#   num_input_channels: Num. channels in prev. layer.
#   filter_size: Width and height of each filter.
#   num_filters: Number of filters.
#   strides: Stride of each dimension
#   padding: Padding
#   use_pooling: Use 2x2 max-pooling.
#   use_batchnorm: Use batch normalisation
#Output:
#   layer: The layer output
#   weights: Weights used by this layer
def new_conv_layer(input,              
                   num_input_channels,
                   filter_size,
                   num_filters,
                   strides=[1,1,1,1],
                   padding='SAME',
                   use_pooling=True,
                   use_batchnorm=False):

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights, for filters with the given shape.
    weights = cnn_util.new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = cnn_util.new_biases(length=num_filters)
    
    # Create the TensorFlow operation for convolution.
    #
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    #
    # Possible values for padding 'SAME', 'VALID'
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=strides,
                         padding=padding)

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Rectified Linear Unit (ReLU). One type of Activation Function
    # Setting values < 0 to 0
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)
    
    # Use pooling to down-sample the image resolution
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        # The setting [1,2,2,1] and padding = 'SAME' is the same as
        # the conv-layer above for the similar reason.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

#Function to reduce the 4-dim tensor (from conv-layer) to 2-dim which can be used as input to the fully-connected layer.
#Input:
#   layer: Input layer assume to be 4-dim
#Output:
#   layer_flat: Output layer assume to be 2-dim
#   num_features: Number of feature
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    
    # layer_shape == [num_images, img_height, img_width, num_channels]
    # The number of features is: img_height * img_width * num_channels
    # From TensorFlow num_elements() is used to calculate num of features.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

#Input:
#   layer: Input layer assume to be 2-dim, flatten
#   num_inputs: Num. inputs from prev. layer.
#   num_outputs: Num. outputs.
#   use_relu: Use Rectified Linear Unit (ReLU)
#Output:
#   layer: Output layer assume to be 2-dim
def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu=True):

    # Create new weights and biases.
    weights = cnn_util.new_weights(shape=[num_inputs, num_outputs])
    biases = cnn_util.new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU to add non-linearity
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
