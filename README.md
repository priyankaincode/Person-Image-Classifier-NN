## PERSON OR NOT IMAGE IDENTIFIER

TWO-LAYER AND L-LAYER BINARY CLASSIFIER MODELS USING NEURAL NETWORKS

**DESCRIPTION** 

This algorithm is able to recognize if an image contains a person/people in it or not. It learns from a set of 1,526 images divided into training, validation, and test sets. The images are stored in an HD5 file, with corresponding labels as to whether they contain a person/people or not.

**INSTALLATION**

This program runs on PYTHON 3 so all dependencies will need to be versions that are compatible with that version.

*Dependencies you will need:*

- Numpy
- Scipy
- Matplotlib
- Python Imaging Library (PIL)

*Image files*

Unfortunately, the images are all too big to be stored in GitHub, so they'll need to be downloaded from [HERE] (). The folder contains the HD5 file already created to map the images. If this doesn't work for you somehow, you can just download the images and then run the array+and+label+data.py file to create your own HD5 data file (which will work with the models so long as they are stored in the same directory).

*Models*

To "install" the models, simply download the Person_or_not_DL.py and pnp_app_utils.py files, and place them in the same directory as the Dataset folder containing the HD5 file and image files.

**USAGE**

*Two-layer model*

The first model included in the program is the two-layer ReLu/sigmoid model with random initialization. To use it, simply comment out everything below line 192. Run Person_or_not_DL.py in the command line to train the model and predict on the validation and test sets.

*L-layer model*

The second model included in the program is the two-layer model with He initialization, L2 regularization, and gradient descent. To use it, comment out the two-layer model training and prediction sections (lines 91 to 191). The L-layer model is flexible and can be edited in line 195 to support different network architectures. The setting I have included here is for a five-layer model, defined as units of 230400, 30, 60, 4, and 1. (Note: the number of input and output layers must remain the same. Only the number and size of hidden layers can be edited.) Run Person_or_not_DL.py in the command line to train the L-layer model and predict on the validation and test sets.

**ANALYSIS**

Based on my testing, the L-layer model shown here - which has five layers (230400, 30, 60, 4, 1) and uses He initialization, L2 regularization, and gradient descent - with a learning rate (alpha) of 0.01 and lambda of 0.8 with 1500 iterations works the best, with **92 percent accuracy on the training set and 81/82 percent accuracy on the validation and test sets.** This could perhaps be improved even more with increased learning rates and decreased lambda to address the overfitting shown in the difference between training and val/test accuracies. I left it at this level of accuracy in the interest of seeing if using TensorFlow would increase accuracy. (Spoiler: it doesn't. :) ) The code for that project can be found [HERE]().

**CREDIT**

I wrote pretty much all of this! But most pieces of code were written originally for various assignments in Dr. Andrew Ng's Deep Learning Specialization courses on Coursera (specifically the Hyperparameter Tuning, Regularization, and Optimization course) so thanks to him and his team for helping me learn to do it all in Python! And for creating the HD5 file, I followed [this tutorial] (http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html) that was immensely helpful as well. And, of course, I would be nowhere without the Stackoverflow community. None of us would be!
