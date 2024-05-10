from PIL import Image
import os
import numpy as np
from testcloud import predict_custom_trained_model_sample
from google.cloud import aiplatform_v1
import time

INPUT_PATH = '../input'

startTime = time.time()

dog_path = ''
for root, dirs, files in os.walk(INPUT_PATH):
    for i in range(len(files)):
        dog_path = root + '/'+files[i]


image = Image.open(dog_path)
resized_image = image.resize((200, 200))

# Convert the resized image to a numpy array
image_array = np.array(resized_image)


# resized_image = image.resize((151, 151), resample=Image.BILINEAR)

image_array = image_array / 255.0  # Normalize pixel values
image_array = image_array.tolist()

instances = {"input_1": image_array}
projectId = "906999370273"
endpoint_id = "8707560345939476480"
predictions = predict_custom_trained_model_sample(
    projectId, endpoint_id, instances)

my_list = []
# Iterate over the elements in the repeated field
for element in predictions[0]:
    # Extract the desired information from each element and append it to the list
    # For example, if the elements are objects with a 'value' field, you can append the values to the list
    my_list.append(float(element))

# Now my_list contains the values extracted from the repeated field
print(my_list)

# Set variables for the current deployed index.
API_ENDPOINT = "466526855.asia-south1-906999370273.vdb.vertexai.goog"
INDEX_ENDPOINT = "projects/906999370273/locations/asia-south1/indexEndpoints/1259600520780185600"
DEPLOYED_INDEX_ID = "dog_face_vector"
# Configure Vector Search client
client_options = {
    "api_endpoint": API_ENDPOINT
}
vector_search_client = aiplatform_v1.MatchServiceClient(
    client_options=client_options,
)

# Build FindNeighborsRequest object
datapoint = aiplatform_v1.IndexDatapoint(
    feature_vector=my_list,
    restricts=[{"namespace": "geoHash", "allow_list": ["tuhsy6"]}]
)
query = aiplatform_v1.FindNeighborsRequest.Query(
    datapoint=datapoint,
    # The number of nearest neighbors to be retrieved
    neighbor_count=5
)
request = aiplatform_v1.FindNeighborsRequest(
    index_endpoint=INDEX_ENDPOINT,
    deployed_index_id=DEPLOYED_INDEX_ID,
    # Request can have multiple queries
    queries=[query],
    return_full_datapoint=False,
)

# Execute the request
response = vector_search_client.find_neighbors(request)
print(response)

neighbors = []
for neighbor in response.nearest_neighbors:
    for neighbor_item in neighbor.neighbors:
        datapoint_id = neighbor_item.datapoint.datapoint_id
        distance = neighbor_item.distance
        neighbor_info = {
            'datapoint_id': datapoint_id,
            'distance': distance
        }
        neighbors.append(neighbor_info)
print(neighbors)
# Handle the response
endTime = time.time()
print("Time-Taken :", endTime-startTime)
# print(response)
