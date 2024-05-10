import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import numpy as np


PATH = '../a/data/after_4_bis/'
# Path to the directory where the history will be stored
PATH_SAVE = '../output/history/'
# Path to the directory where the model will be stored
PATH_MODEL = '../output/model/2024.03.16/'
SIZE = (151, 151, 3)                               # Size of the input images
TEST_SPLIT = 0.1                                       # Train/test ratio

# Load a network from a saved model? If True NET_NAME and START_EPOCH have to be precised
LOAD_NET = True
NET_NAME = '2019.07.29.dogfacenet'                   # Network saved name
# Start the training at a specified epoch
START_EPOCH = 51
# Number of epoch to train the network
NBOF_EPOCHS = 60
# Use high level training ('fit' keras method)
HIGH_LEVEL = True
# Number of steps per epoch
STEPS_PER_EPOCH = 300
# Number of steps per validation
VALIDATION_STEPS = 30


alpha = 0.3


def triplet(y_true, y_pred):

    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]

    ap = K.sum(K.square(a-p), -1)
    an = K.sum(K.square(a-n), -1)

    return K.sum(tf.nn.relu(ap - an + alpha))


def triplet_acc(y_true, y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]

    ap = K.sum(K.square(a-p), -1)
    an = K.sum(K.square(a-n), -1)

    return K.less(ap+alpha, an)

# -----


model = tf.keras.models.load_model(
    '{:s}{:s}.{:d}.h5'.format(PATH_MODEL, NET_NAME, START_EPOCH),
    custom_objects={'triplet': triplet, 'triplet_acc': triplet_acc})

print(model.summary())
input_layer_name = model.input.name
print("Input layer name:", input_layer_name)

