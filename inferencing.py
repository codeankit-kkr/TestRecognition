from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="J6TtEppk30UrtxpzIj4r"
)

result = CLIENT.infer('./../musky.4.jpg', model_id="dog-breed-xpaq6/1")
print(result['predictions'][0]['class'].split('.')[1])
