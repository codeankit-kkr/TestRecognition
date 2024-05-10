"""
Evaluation of DogFaceNet model

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.image as mpimg
import heapq

import tensorflow as tf

import os
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from online_training2 import *

# ----------------------------------------------------------------------------
# Config.

# Path to the directory of the saved dataset
PATH = 'C:/Users/HP/Downloads/DogFaceNet_Dataset_224_1/after_4_bis/'
# Path to the directory where the history will be stored
PATH_SAVE = '../output/history/'
# Path to the directory where the model will be stored
PATH_MODEL = '../output/model/2024.05.03/'
INPUT_PATH = '../input'
SIZE = (224, 224, 3)                               # Size of the input images
TEST_SPLIT = 0.1                              # Train/test ratio

NET_NAME = '2024.05.03.dogfacenet'               # Network saved name
START_EPOCH = 237             # Start the training at a specified epoch

# ----------------------------------------------------------------------------
# Import the dataset.

assert os.path.isdir(PATH), '[Error] Provided PATH for dataset does not exist.'

print('Loading the dataset...')

filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root, dirs, files in os.walk(PATH):
    if len(files) > 1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
        filenames = np.append(filenames, files)
        labels = np.append(labels, np.ones(len(files))*idx)
        idx += 1
assert len(labels) != 0, '[Error] No data provided.'

print('Done.')

print('Total number of imported pictures: {:d}'.format(len(labels)))

nbof_classes = len(np.unique(labels))
print('Total number of classes: {:d}'.format(nbof_classes))

lost_dog = []
for root, dirs, files in os.walk(INPUT_PATH):
    for i in range(len(files)):
        lost_dog = lost_dog + [root + '/'+files[i]]

# ----------------------------------------------------------------------------
# Split the dataset.

# nbof_test = int(TEST_SPLIT*nbof_classes)

# keep_test = np.less(labels,nbof_test)
# keep_train = np.logical_not(keep_test)

# filenames_test = filenames[keep_test]
# labels_test = labels[keep_test]

# filenames_train = filenames[keep_train]
# labels_train = labels[keep_train]

# print("Number of training data: " + str(len(filenames_train)))
# print("Number of training classes: " + str(nbof_classes-nbof_test))
# print("Number of testing data: " + str(len(filenames_test)))
# print("Number of testing classes: " + str(nbof_test))

# #----------------------------------------------------------------------------
# # Loss definition.

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


print('Loading model from {:s}{:s}.{:d}.h5 ...'.format(
    PATH_MODEL, NET_NAME, START_EPOCH))

model = tf.keras.models.load_model(
    '{:s}{:s}.{:d}'.format(PATH_MODEL, NET_NAME, START_EPOCH),
    custom_objects={'triplet': triplet, 'triplet_acc': triplet_acc})

print('Done.')

# #----------------------------------------------------------------------------
# # Verification task, create pairs

print('Verification task, pairs creation...')

# NBOF_PAIRS = 5000
# #NBOF_PAIRS = len(images_test)

# # Create pairs
# h,w,c = SIZE
# pairs = []
# class_test = np.unique(labels_test)
# for i in range(NBOF_PAIRS):

#         class1 = np.random.randint(len(class_test))

#         images_class1 = filenames_test[np.equal(labels_test,class1)]

#         # Chose an image amoung these selected images
#         pairs = pairs + [images_class1[np.random.randint(len(images_class1))]]
#         pairs = pairs + ["C:/Users/HP/Downloads/mydog21.jpg"]


# print('Done.')

# #----------------------------------------------------------------------------
# # Verification task, evaluate the pairs

print('Verification task, model evaluation...')

predict_all = model.predict(predict_generator(
    filenames, 32), steps=np.ceil(len(filenames)/32))
predict_lost = model.predict(predict_generator(
    lost_dog, 32), steps=np.ceil(len(lost_dog)/32))
print(predict_lost)

diff = np.square(predict_all - predict_lost[0])
dist = np.sum(diff, 1)


# # Separates the pairs
# emb1 = predict[0::2]
# emb2 = predict[1::2]

# # Computes distance between pairs
# diff = np.square(emb1-emb2)
# dist = np.sum(diff,1)

# # print("Euclidean Distance ",dist)
# # best = 0
# # best_t = 0
# # thresholds = np.arange(0.001,4,0.001)
# # for i in range(len(thresholds)):
# #     less = np.less(dist, thresholds[i])
# #     acc = np.logical_not(np.logical_xor(less, issame))
# #     acc = acc.astype(float)
# #     out = np.sum(acc)
# #     out = out/len(acc)
# #     if out > best:
# #         best_t = thresholds[i]
# #         best = out

# # print('Done.')
# # print("Best threshold: " + str(best_t))
# # print("Best accuracy: " + str(best))

# # Test: Look at wrong pairs


def pair_compare(pair1, pair2):
    return pair1[0] < pair2[0]


min_heap = []

best = float('inf')
ind = 10
threshold = 0.9
for i in range(len(dist)):
    if dist[i] < best and dist[i] < threshold:
        best = dist[i]
        ind = i
        heapq.heappush(min_heap, [dist[i], i])

print(ind)
# print(best)
# print(2*ind+1)


# # s = 10
# # sr = 20
# # n = 5
# # print('Ground truth: {:s}'.format(str(issame[s:(n+s)])))
# # fig = plt.figure(figsize=(11,2.8*5))
# # for i in range(s,s+n):
# #     # False accepted: columns 1 and 2
# #     plt.subplot(n,4,4*(i-s)+1)
# #     plt.imshow(load_images([pairs[2*fa[i+s]]])[0])
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.subplot(n,4,4*(i-s)+2)
# #     plt.imshow(load_images([pairs[2*fa[i+s]+1]])[0])
# #     plt.xticks([])
# #     plt.yticks([])
# #     # False rejected: columns 3 and 4
# #     plt.subplot(n,4,4*(i-s)+3)
# #     plt.imshow(load_images([pairs[2*fr[i+sr]]])[0])
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.subplot(n,4,4*(i-s)+4)
# #     plt.imshow(load_images([pairs[2*fr[i+sr]+1]])[0])
# #     plt.xticks([])
# #     plt.yticks([])

# # import matplotlib.pyplot as plt
temp = 5
image_paths = []
while (min_heap):
    min_element = heapq.heappop(min_heap)
    image_paths = image_paths + [filenames[min_element[1]]]
    print(min_element)
    temp = temp-1


def show_images_from_paths(image_paths):
    # Limit to at most 5 subplots
    fig, axes = plt.subplots(1, min(len(image_paths), 10), figsize=(15, 3))

    # If there are no images, just return
    if len(image_paths) == 0:
        print("No images to display.")
        return

    # If there is only one image, axes will not be iterable, so handle it separately
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Iterate over the axes and images
    for i, ax in enumerate(axes):
        image = mpimg.imread(image_paths[i])
        ax.imshow(image)
        ax.set_title(f"{os.path.basename(image_paths[i])}")
        ax.axis('off')

    plt.show()


# Example usage
# Assuming 'image_paths' is a list containing paths to your images
show_images_from_paths(image_paths)

# def show_single_image(image_path):
#     img = mpimg.imread(image_path)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()

# print(filenames[ind])
# show_single_image(filenames[ind])
# # Example usage:
# # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]  # Replace these with your actual image paths


# # plt.show()

# # threshold = 0.3
# # less = np.less(dist, threshold)
# # acc = np.logical_not(np.logical_xor(less, issame))
# # acc = acc.astype(float)
# # out = np.sum(acc)
# # out = out/len(acc)

# # print("Accuracy: " + str(out))
