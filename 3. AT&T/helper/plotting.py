import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

class Plotting(object):
    def __init__(self, num_classes, num_channels, class_names):
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.class_names = class_names
        
    #Function print the confusion matrix of testset true class against 
    #predicted classes.
    #Input:
    #   test_cls: True class of test-set.
    #   cls_pred: Predicted class
    #Output:
    #   None
    def plot_confusion_matrix(self, test_cls, cls_pred):
        # This is called from print_test_accuracy() below.

        # cls_pred is an array of the predicted class-number for
        # all images in the test-set.

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=test_cls,
                              y_pred=cls_pred)

        # Print the confusion matrix as text.
        for i in range(self.num_classes):
            # Append the class-name to each line.
            classname = "({}) {}".format(i, self.class_names[i])
            print(cm[i, :], classname)

        # Print the class-numbers for easy reference.
        class_numbers = [" ({0})".format(i) for i in range(self.num_classes)]
        print("".join(class_numbers))
    
    #Function print examples of some mis-classified images.
    #This function assume using test-set
    #Input:
    #   cls_pred: Predicted class for each image
    #   correct: boolean array, whether predicted class equal true class for each image
    #Output:
    #   None
    def plot_example_errors(self, test_image, test_cls, cls_pred, correct):
        # Negate the boolean array. To get the incorrect images.
        incorrect = (correct == False)

        # Get the images from the test-set that have been
        # incorrectly classified.
        images = test_image[incorrect]

        # Get the predicted classes for those images.
        cls_pred = cls_pred[incorrect]

        # Get the true classes for those images.
        cls_true = test_cls[incorrect]

        # Plot the first 9 images.
        self.plot_images(images=images[0:9],
                         cls_true=cls_true[0:9],
                         cls_pred=cls_pred[0:9])
        
    def plot_images(self, images, cls_true, cls_pred=None, smooth=True):
        assert len(images) == len(cls_true) == 9

        # Create figure with sub-plots.
        fig, axes = plt.subplots(3, 3)

        # Adjust vertical spacing if we need to print ensemble and best-net.
        if cls_pred is None:
            hspace = 0.3
        else:
            hspace = 0.6
            fig.subplots_adjust(hspace=hspace, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Interpolation type.
            if smooth:
                interpolation = 'spline16'
            else:
                interpolation = 'nearest'

            # Plot image.
            if self.num_channels == 1:
                ax.imshow(images[i, :, :, 0], interpolation=interpolation, cmap='gray')
            else:
                ax.imshow(images[i, :, :, :], interpolation=interpolation, cmap='gray')

            # Name of the true class.
            cls_true_name = self.class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = self.class_names[cls_pred[i]]    
                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()
    