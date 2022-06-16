import requests
import json
import base64
import numpy as np
from io import BytesIO

url = "https://square.ukp-lab.de/auth/realms/square/protocol/openid-connect/token"

payload='grant_type=client_credentials&client_id=models&client_secret=2mNXNJJHysAmL8RV6GIAuotwQ6eDfkkt'
headers = {
  'Content-Type': 'application/x-www-form-urlencoded'
}

response = requests.request("POST", url, headers=headers, data=payload)

# print(type(response.text))
# print(response.json()['access_token'])
# print('done')
current_token ='Bearer '+ response.json()['access_token']
# print(current_token)

url = "https://square.ukp-lab.de/api/facebook-dpr-question_encoder-single-nq-base/embedding"

payload = json.dumps({
  "input": [
    "How are you!"
  ],
  "is_preprocessed": False,
  "preprocessing_kwargs": {},
  "model_kwargs": {
    "output_hidden_states": True
  },
  "task_kwargs": {},
  "adapter_name": ""
})
headers = {
  'accept': 'application/json',
  'Authorization': current_token,
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

# print(response.json()['model_outputs']['pooler_output'])
pooler_output = response.json()['model_outputs']['pooler_output']
#print(pooler_output.encode())
print(base64.decodebytes(pooler_output.encode()))



decoded_array = np.load(BytesIO( base64.decodebytes(pooler_output.encode())) )

print( decoded_array.shape )
#arr_binary = b.getvalue()
# arr_binary_b64 = base64.b64encode(arr_binary)
# arr_string_b64 = arr_binary_b64.decode("latin1")
# # return arr_string_b64
#
# # DECODE THE VALUE WITH
# arr_binary_b64 = arr_string_b64.encode()
# arr_binary = base64.decodebytes(arr_binary_b64)
# arr = np.load(BytesIO(arr_binary))
#
# data = {
#       "input": [
#         "Do aliens exist?"
#           ],
#       "is_preprocessed": "False",
#       "preprocessing_kwargs": {},
#       "model_kwargs": {},
#       "task_kwargs": {},
#       "adapter_name": ""
#     }
#     client = ManagementClient(client_secret="2mNXNJJHysAmL8RV6GIAuotwQ6eDfkkt",
#                               api_url="https://localhost:8443",
#                               verify_ssl=False)
#     result = client.predict(model_identifier="facebook-dpr-question_encoder-single-nq-base",
#                             prediction_method="embedding",
#                             input_data=data)
#     print(result)
