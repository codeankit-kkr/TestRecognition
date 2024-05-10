import functions_framework
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from typing import Dict, List, Union
from google.cloud import aiplatform_v1
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from inference_sdk import InferenceHTTPClient
import json


@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response JSON object.
    """
    projectId = "906999370273"
    endpoint_id = "4966546799222325248"
    vectorIndexId = "507077170544246784"
    request_json = request.get_json(silent=True)
    if not request_json:
        return json.dumps({'message': "No JSON payload in request"}), 400, {'Content-Type': 'application/json'}

    if 'updateDatapoint' in request_json:
        if 'base64Arr' in request_json:
            base64Arr = request_json['base64Arr']
        else:
            return json.dumps({'message': "No base64arr in request"}), 400, {'Content-Type': 'application/json'}
        if 'reqGeoHash' in request_json:
            reqGeoHash = request_json['reqGeoHash']
        else:
            return json.dumps({'message': "No Geohash in request"}), 400, {'Content-Type': 'application/json'}
        if 'petId' in request_json:
            petId = request_json['petId']
        else:
            return json.dumps({'message': "No petId in request"}), 400, {'Content-Type': 'application/json'}
        try:
            allDatapoints = []
            for imageBase64 in base64Arr:
                img_data = base64.b64decode(imageBase64)
                img = Image.open(BytesIO(img_data))
                img = img.resize((151, 151))
                image = np.array(img)
                image = image / 255.0
                if image.shape[-1] != 3:
                    # Convert grayscale to RGB
                    image = np.stack((image,) * 3, axis=-1)
                image = image.tolist()
                instances = {"input_1": image}
                predictions = predict_custom_trained_model_sample(
                    projectId, endpoint_id, instances)
                if (len(predictions) > 0):
                    allDatapoints.append(predictions[0])
            print("prediction completed")
            datapoints = []
            print("all Datapoints :", allDatapoints)
            for feature_vector in allDatapoints:
                print("petId:", petId)
                print("feature_vector:", feature_vector)
                print("GeoHash:", reqGeoHash)
                obj = aiplatform_v1.IndexDatapoint(
                    datapoint_id=petId,
                    feature_vector=feature_vector,
                    restricts=[{"namespace": "geoHash", "allow_list": [reqGeoHash]}])
                print(obj)
                datapoints.append(obj)
                print(datapoints)
            print("datapoints created")
            stream_update_vector_search_index(
                projectId, "asia-south1", vectorIndexId, datapoints)
            return json.dumps({'message': "Successfully updated vector index"}), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            return json.dumps({'message': e, }), 400, {'Content-Type': 'application/json'}

    if 'base64Img' in request_json:
        base64_img = request_json['base64Img']
    else:
        return 'Error: base64Img not found in request', 400
    if 'reqGeoHash' in request_json:
        reqGeoHash = request_json['reqGeoHash']
    else:
        return 'Error: reqGeoHash not found in request', 400

    breed = "Unknown"
    try:
        img_data = base64_img
        CLIENT = InferenceHTTPClient(
            api_url="https://classify.roboflow.com",
            api_key="J6TtEppk30UrtxpzIj4r"
        )

        result = CLIENT.infer(base64_img, model_id="dog-breed-xpaq6/1")
        if ('predictions' in result):
            breed = result['predictions'][0]['class'].split('.')[1]
        img_data = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_data))
    except Exception as e:
        return json.dumps({'message': "Error decoding base64 image"}), 400, {'Content-Type': 'application/json'}

    # Log the image dimensions
    img = img.resize((151, 151))
    image = np.array(img)
    image = image / 255.0
    # image = np.expand_dims(image, axis=0)  # Add
    if image.shape[-1] != 3:
        # Convert grayscale to RGB
        image = np.stack((image,) * 3, axis=-1)
    image = image.tolist()
    instances = {"input_1": image}
    predictions = predict_custom_trained_model_sample(
        projectId, endpoint_id, instances)

    my_list = []
    # Iterate over the elements in the repeated field
    for element in predictions[0]:
        # Extract the desired information from each element and append it to the list
        # For example, if the elements are objects with a 'value' field, you can append the values to the list
        my_list.append(float(element))

    # Now my_list contains the values extracted from the repeated field
    # print(my_list)

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
        restricts=[{"namespace": "geoHash", "allow_list": [reqGeoHash]}]
    )
    query = aiplatform_v1.FindNeighborsRequest.Query(
        datapoint=datapoint,

        # The number of nearest neighbors to be retrieved
        neighbor_count=10
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

# Return a response
    response_data = {'nearest_neighbors': neighbors, 'breed': breed}
    return json.dumps(response_data), 200, {'Content-Type': 'application/json'}


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "asia-south1",
    api_endpoint: str = "asia-south1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    # print(endpoint)
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    return predictions


def stream_update_vector_search_index(
    project: str, location: str, index_name: str, datapoints
) -> None:

    aiplatform.init(project=project, location=location)

    my_index = aiplatform.MatchingEngineIndex(index_name=index_name)

    my_index.upsert_datapoints(datapoints=datapoints)
