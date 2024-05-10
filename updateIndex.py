from google.cloud import aiplatform
import os
import numpy as np
import json
import tensorflow as tf
from PIL import Image

PATH = '../a/data/incremental/'  # Path to the directory of the saved dataset
# Path to the directory where the model will be stored
PATH_MODEL = '../output/model/2024.03.28/'

NET_NAME = 'dogfacenet_model_'  # Network saved name
START_EPOCH = 7

alpha = 0.3


def triplet(y_true, y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]

    ap = tf.keras.backend.sum(tf.keras.backend.square(a-p), -1)
    an = tf.keras.backend.sum(tf.keras.backend.square(a-n), -1)

    return tf.keras.backend.sum(tf.nn.relu(ap - an + alpha))


def triplet_acc(y_true, y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]

    ap = tf.keras.backend.sum(tf.keras.backend.square(a-p), -1)
    an = tf.keras.backend.sum(tf.keras.backend.square(a-n), -1)

    return tf.keras.backend.less(ap+alpha, an)


print('Loading model from {:s}{:s}{:d} ...'.format(
    PATH_MODEL, NET_NAME, START_EPOCH))

model = tf.keras.models.load_model(
    '{:s}{:s}{:d}/'.format(PATH_MODEL, NET_NAME, START_EPOCH),
    custom_objects={'triplet': triplet, 'triplet_acc': triplet_acc})

print('Done.')
print(model.summary())

# Function to generate embeddings for images


def generate_embeddings(image_path):
    image = np.array(Image.open(image_path))
    image = image / 255.0  # Normalize pixel values
    img_array = np.expand_dims(image, axis=0)
    embeddings = model.predict(img_array)
    return embeddings.flatten().tolist()  # Convert to list for JSON serialization


with open("incrementalData.json", "a") as f:
    for root, dirs, files in os.walk(PATH):
        for dog_folder in dirs:
            dog_folder_path = os.path.join(root, dog_folder)
            for img_file in os.listdir(dog_folder_path):
                img_path = os.path.join(dog_folder_path, img_file)
                embedding = generate_embeddings(img_path)
                # Extract filename without extension
                encoded_name = img_file.split('.')[0]
                f.write('{"id":"' + str(encoded_name) + '",')
                f.write('"embedding":[' + ",".join(str(x)
                        for x in embedding)+'],')
                f.write(
                    '"restricts":['+'{"namespace":"geoHash","allow":["9q9hvumpq"]}]'+'}')
                f.write("\n")


BUCKET_URI = "gs://mtm-dog-face-recognition"
aiplatform.init(project="innate-agency-377608",
                location="asia-south1", staging_bucket=BUCKET_URI)
INDEX_RESOURCE_NAME = "500603246079901696"
tree_ah_index = aiplatform.MatchingEngineIndex(index_name=INDEX_RESOURCE_NAME)
EMBEDDINGS_UPDATE_URI = f"{BUCKET_URI}/incrementalVectors/"
tree_ah_index = tree_ah_index.update_embeddings(
    contents_delta_uri=EMBEDDINGS_UPDATE_URI,
)
