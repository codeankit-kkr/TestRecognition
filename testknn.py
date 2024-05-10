import tensorflow as tf

import os
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from PIL import Image
import pickle
import time


PATH = '../a/data/test_ruffles/'  # Path to the directory of the saved dataset
# Path to the directory where the history will be stored
PATH_SAVE = '../output/history/'
# Path to the directory where the model will be stored
PATH_MODEL = '../output/model/2024.03.16/'
INPUT_PATH = '../input'
SIZE = (151, 151, 3)                               # Size of the input images
TEST_SPLIT = 0.1                              # Train/test ratio
OUTPUT_PATH = "artefacts"

NET_NAME = '2019.07.29.dogfacenet'               # Network saved name
START_EPOCH = 51             # Start the training at a specified epoch

lost_dog = ''
for root, dirs, files in os.walk(INPUT_PATH):
    for i in range(len(files)):
        lost_dog = root + '/'+files[i]

print(lost_dog)

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


# #----------------------------------------------------------------------------
# # Model definition.
start = time.time()

print('Loading model from {:s}{:s}.{:d}.h5 ...'.format(
    PATH_MODEL, NET_NAME, START_EPOCH))

model = tf.keras.models.load_model(
    '{:s}{:s}.{:d}.h5'.format(PATH_MODEL, NET_NAME, START_EPOCH),
    custom_objects={'triplet': triplet, 'triplet_acc': triplet_acc})

print('Done.')


with open(os.path.join(OUTPUT_PATH, "knn.pkl"), 'rb') as knn_file:
    knn = pickle.load(knn_file)

with open(os.path.join(OUTPUT_PATH, "lEncoder.pkl"), 'rb') as encoder_file:
    le = pickle.load(encoder_file)


def query_image(image_path):
    # Load and preprocess the image
    image = np.array(Image.open(image_path))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    # Get the image embedding using the TensorFlow model
    q_embedding = model.predict(image)

    # Get the predicted probabilities from the KNN classifier
    ranked = np.argsort(knn.predict_proba(q_embedding)[0])[::-1]
    # Inverse transform the predicted labels using the label encoder
    ranked_labels = le.inverse_transform(ranked)[:4]

    # Filter out label '999' and return top 4 ranked labels
    ranked_labels = [label for label in ranked_labels]
    print(ranked_labels)
    return ranked_labels


dogs_found = query_image(lost_dog)
end = time.time()
print("Time Taken:", end-start)

fig, axes = plt.subplots(nrows=(len(dogs_found)+1), ncols=5, figsize=(20, 8))

input_image = Image.open(lost_dog)
axes[0, 0].imshow(input_image)
axes[0, 0].set_title("Input Image")
axes[0, 0].axis("off")

# Iterate through each folder and plot images in the second row
for j, folder_name in enumerate(dogs_found):
    folder_path = os.path.join(PATH, folder_name)

    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Plot images from each folder
    for i, file_name in enumerate(files[:5]):  # Plot up to 8 images per folder
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path)
        axes[j+1, i].imshow(image)
        axes[j+1, i].set_title(f"{folder_name}")
        axes[j+1, i].axis("off")

# Adjust layout to prevent overlapping
plt.tight_layout()


# Show the plot
plt.show()
