import ipdb
import requests
import json
from easy_conn import get_token, base_url

import argparse
# base_url = 
# # print(base_url)
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
args = parser.parse_args()



url = "https://square.ukp.informatik.tu-darmstadt.de/auth/realms/square/protocol/openid-connect/token"

payload='grant_type=password&client_id=web-app&username=haoz&password=VQwB9EPF8qYpeux'
headers = {
  'Content-Type': 'application/x-www-form-urlencoded'
}

response = requests.request("POST", url, headers=headers, data=payload)




sec = response.json()['access_token']


# datastore_name = 'conceptnet'
datastore_name = args.name
#ipdb.set_trace()
response = requests.put(
    f"{'http://127.0.0.1:7000'}/datastores/kg/{datastore_name}", 
    headers={
        "Authorization": f"Bearer {sec}"
    },
    json=[
      {
        "name": "name",
        # "type": "text"
        "type": "keyword"
      },
      {
        "name": "type",  # ['node', 'edge']
        "type": "keyword"
      },
      {
        "name": "description",
        "type": "text"
      },
      {
        "name": "weight",
        "type": "double"
      },

      # these two will be empty for nodes:
      {
        "name": "in_id",
        "type": "keyword"
      },
      {
        "name": "out_id",
        "type": "keyword"
      },
      # for nodes?
      # {
      #   "name": "in_relation",
      #   "type": "text"
      # },
      # {
      #   "name": "out_relation",
      #   "type": "text"
      # }
    ]
)
print(response.status_code)
print(response.json())

# response = requests.get(
#     "http://localhost:7000/datastores", 
#     headers={
#         "Authorization": f"Bearer {get_token()}"
#     }
# )
# print(json.dumps(response.json(), indent=4))