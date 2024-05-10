import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import tensorflow.keras.backend as K
import pickle
from PIL import Image
from glob import glob
import os
import numpy as np

# Constants
INDEXING_PATH = "../a/data/test_ruffles/**/*.jpg"
OUTPUT_PATH = "artefacts"
PATH_MODEL  = '../output/model/2024.03.16/'     
NET_NAME    = '2019.07.29.dogfacenet'                   # Network saved name
START_EPOCH = 51    

# Load model
alpha = 0.3
def triplet(y_true,y_pred):
    
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)

    return K.sum(tf.nn.relu(ap - an + alpha))

def triplet_acc(y_true,y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)
    
    return K.less(ap+alpha,an)


def load_model():
    print('Loading model from {:s}{:s}.{:d}.h5 ...'.format(PATH_MODEL,NET_NAME,START_EPOCH))

    model = tf.keras.models.load_model(
        '{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,START_EPOCH),
        custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})
    
    print('Done.')
    return model

# Define DogIndexingDataset
class DogIndexingDataset(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size=32):
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = [np.array(Image.open(path)) for path in batch_paths]
        batch_images = np.array(batch_images) / 255.0  # Normalize pixel values
        return batch_images, batch_labels

# Function to generate index labels
def gen_index_labels():
    indexed_files = glob(INDEXING_PATH)
    label_index = [t.split("\\")[-1].split(".")[0] for t in indexed_files]
    le = LabelEncoder()
    label_ = le.fit_transform(label_index)
    return indexed_files, label_index, le, label_

if __name__ == "__main__":
    # Load model
    model = load_model()

    # Load data
    indexed_files, label_index, le, label_ = gen_index_labels()

    # Generate embeddings
    embeddings = []
    labels = []
    batch_size = 30  # Adjust batch size as needed
    for i in range(0, len(indexed_files), batch_size):
        batch_paths = indexed_files[i:i + batch_size]
        batch_labels = label_index[i:i + batch_size]
        batch_images = [np.array(Image.open(path)) for path in batch_paths]
        batch_images = np.array(batch_images)/255.0 # Normalize pixel values
        batch_embeddings = model.predict(batch_images)
        embeddings.extend(batch_embeddings)
        labels.extend(batch_labels)

    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3,metric='minkowski')
    knn.fit(embeddings, labels)
    # print(embeddings)
    # print(labels)

    # Save artefacts
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(os.path.join(OUTPUT_PATH, "knn.pkl"), 'wb') as knn_file:
        pickle.dump(knn, knn_file)

    with open(os.path.join(OUTPUT_PATH, "lEncoder.pkl"), 'wb') as encoder_file:
        pickle.dump(le, encoder_file)
