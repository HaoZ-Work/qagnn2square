'''
This file is for using API in QAGNN model.

'''
import requests
import json
import base64
import numpy as np
from io import BytesIO
import time
import pprint
from itertools import product
KG_NAME = "conceptnet"
# BASE_URL = 'http://172.29.101.165:7000'
BASE_URL ='https://square.ukp-lab.de/api'
HOPS = 2


#url = "https://square.ukp-lab.de/api/facebook-dpr-question_encoder-single-nq-base/embedding"
#
# payload = json.dumps({
#   "input": [
#     "How are you!"
#   ],
#   "is_preprocessed": False,
#   "preprocessing_kwargs": {},
#   "model_kwargs": {
#     "output_hidden_states": True
#   },
#   "task_kwargs": {},
#   "adapter_name": ""
# })
# headers = {
#   'accept': 'application/json',
#   'Authorization': current_token,
#   'Content-Type': 'application/json'
# }
#
# response = requests.request("POST", url, headers=headers, data=payload)
#
# # print(response.json()['model_outputs']['pooler_output'])
# pooler_output = response.json()['model_outputs']['pooler_output']
# #print(pooler_output.encode())
# print(base64.decodebytes(pooler_output.encode()))
#
#
#
# decoded_array = np.load(BytesIO( base64.decodebytes(pooler_output.encode())) )
#
# print( decoded_array.shape )
# #arr_binary = b.getvalue()
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


##  Get node by id



# def auth():
#   '''
#   Get the autho token from UKP server
#
#   Args:
#
#   Return:
#     Token
#
#   '''
#   auth_url = "https://square.ukp.informatik.tu-darmstadt.de/auth/realms/square/protocol/openid-connect/token"
#   auth_payload = 'grant_type=client_credentials&client_id=models&client_secret=2mNXNJJHysAmL8RV6GIAuotwQ6eDfkkt'
#   auth_headers = {
#     'Content-Type': 'application/x-www-form-urlencoded'
#   }
#   auth_response = requests.request("POST", auth_url, headers=auth_headers, data=auth_payload)
#   #auth_sec = auth_response.json()['access_token']
#
#   return auth_response.json()['access_token']


def auth():
  '''
  Get the autho token from UKP server

  Args:

  Return:
    Token

  '''
  auth_url = "https://square.ukp-lab.de/auth/realms/square/protocol/openid-connect/token"
  auth_payload = 'grant_type=password&client_id=web-app&username=haoz&password=VQwB9EPF8qYpeux'
  auth_headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
  }
  auth_response = requests.request("POST", auth_url, headers=auth_headers, data=auth_payload)
  #auth_sec = auth_response.json()['access_token']
  # print(auth_response.json()['access_token'])
  return auth_response.json()['access_token']

AUTH_SEC = auth()

def concept2id_api(node_name:str):
  '''
  Get the node id by given node name.

  Args:
    node_name: str, the name of node
  Return:
    node id: int, e.g. 1271

  '''
  AUTH_SEC = auth()
  response = requests.post(
    f"{BASE_URL}/datastores/kg/{KG_NAME}/nodes/query_by_name",
    headers={
        "Authorization": f"Bearer {AUTH_SEC}"
    },
    data = json.dumps([node_name])

  )

  #print(nid_to_int(response.json()[0][0]))
  #1271

  #return np.random.randint(1,3287,size=1)[0]
  return nid_to_int(response.json()[0][0])


def id2concept_api(node_id:int):
  '''
  Get the name of node by give id

  Args:
    node_id: int, the

  Return:
    node_name: str

  '''
  global AUTH_SEC
  response = requests.get(
    f"{BASE_URL}/datastores/kg/{KG_NAME}/{nid_to_str(node_id)}",
    headers={
      "Authorization": f"Bearer {AUTH_SEC}"
    }
  )
  # print(response.json())
  if response.status_code != 200:
    AUTH_SEC = auth()
    response = requests.get(
      f"{BASE_URL}/datastores/kg/{KG_NAME}/{nid_to_str(node_id)}",
      headers={
        "Authorization": f"Bearer {AUTH_SEC}"
      }
    )
  return response.json()[list(response.json().keys())[0]]['name']


def cpnet_simple_api(node_id:int):
  '''
  Get the id of the nodes which has HOPS to the given node.

  Args:
      node_id : int,
  Return:
    List: list of node ids


  '''
  global AUTH_SEC
  #print(f'node id:{node_id}')
  response = requests.get(
      f"{BASE_URL}/datastores/kg/{KG_NAME}/{nid_to_str(node_id)}",
      headers={
          "Authorization": f"Bearer {AUTH_SEC}"
      }
  )
  if response.status_code != 200:
    AUTH_SEC = auth()
    response = requests.get(
      f"{BASE_URL}/datastores/kg/{KG_NAME}/{nid_to_str(node_id)}",
      headers={
        "Authorization": f"Bearer {AUTH_SEC}"
      }
    )

  #print(response.json())

  node_name = response.json()[list(response.json().keys())[0]]['name']



  #print(node_name)

  #print(f'node name:{node_name}')
  try:
    response = requests.post(
      f"{BASE_URL}/datastores/kg/{KG_NAME}/subgraph/query_by_node_name",
      headers={
        "Authorization": f"Bearer {AUTH_SEC}"
      },
      json={
        "nids": [node_name],
        "hops": HOPS
      }
    )
    if response.status_code != 200:
      AUTH_SEC = auth()
      response = requests.post(
        f"{BASE_URL}/datastores/kg/{KG_NAME}/subgraph/query_by_node_name",
        headers={
          "Authorization": f"Bearer {AUTH_SEC}"
        },
        json={
          "nids": [node_name],
          "hops": HOPS
        }
      )
    return [nid_to_int(i) for i in list(response.json()[0].keys())]
  except (requests.exceptions.JSONDecodeError,json.decoder.JSONDecodeError):
    # if this node is not in subgraph,return False
    return False
  #print(list(response.json()[0].keys()))
  else:
    return [nid_to_int(i) for i in list(response.json()[0].keys())]

def nid_to_int(node_id: str):
    '''
    Convert node id for string to int
        'n12345'--> 12345

    Args:
        node_id: str , e.g.

    Return:
        node id : int

    '''
    node_id = node_id.replace('n', '')
    # print(node_id)

    return int(node_id)


def nid_to_str(node_id: int):
  '''
  Convert node id for int to string
  12345-->'n12345'


  Args:
      node_id: int

  Return:
      node id : str

  '''
  node_id = 'n' + str(node_id)
  #print(node_id)

  return node_id

def merged_relations_api():
  '''

  Return: a list of relations' names

  '''
  response = requests.get(
    f"{BASE_URL}/datastores/kg/{KG_NAME}/relations",
    headers={
      "Authorization": f"Bearer {AUTH_SEC}"
    }
  )
  # print(response.status_code)

  return [i['key'] for i in response.json()['name']['buckets']]

id2relation_api = merged_relations_api()
relation2id_api = {r: i for i, r in enumerate(id2relation_api)}

def cpnet_has_edge(node_1:int, node_2:int):
  '''
  Check whether there is a edge between node1 and node2

  Args:
      node_1,node_2: int, node id

  Return:
    True: when there is a edge
    False: there is no edge or the node is not in graph
  '''
  re = cpnet_simple_api(node_1)
  if re != False:
    if node_2 in re:
      return True
    else:
      return False
  else:
    return False

def cpnet_values(node_1:int, node_2:int):
  '''
  Get the weight value and relation id of a edge by given 2 nodes

  Args:
      node_1,node_2: int, node id

  Return:
    dict: {'rel': int rel_id,'weight':float weight}

  '''

  # print(f'node id:{node_id}')
  global AUTH_SEC
  response = requests.get(
    f"{BASE_URL}/datastores/kg/{KG_NAME}/{nid_to_str(node_1)}",
    headers={
      "Authorization": f"Bearer {AUTH_SEC}"
    }
  )
  if response.status_code != 200:
    AUTH_SEC = auth()
    response = requests.get(
      f"{BASE_URL}/datastores/kg/{KG_NAME}/{nid_to_str(node_1)}",
      headers={
        "Authorization": f"Bearer {AUTH_SEC}"
      }
    )
  # print(response.json())

  node_name = response.json()[list(response.json().keys())[0]]['name']
  # print(node_name)

  # print(f'node name:{node_name}')
  try:
    response = requests.post(
      f"{BASE_URL}/datastores/kg/{KG_NAME}/subgraph/query_by_node_name",
      headers={
        "Authorization": f"Bearer {AUTH_SEC}"
      },
      json={
        "nids": [node_name],
        "hops": HOPS
      }
    )
    re = dict()
    for i in response.json()[1].values():
      if nid_to_str(node_1)==i['in_id'] and nid_to_str(node_2) == i['out_id']:
        # print(dict({'rel':relation2id_api[i['name']],'weight':i['weight']}))
        re[len(re)]= dict({'rel':relation2id_api[i['name']],'weight':i['weight']})
    return re

  except (requests.exceptions.JSONDecodeError, json.decoder.JSONDecodeError):
    # if this node is not in subgraph,return False
    return dict()
  # print(list(response.json()[0].keys()))
  # else:
  #   re = dict()
  #   for i in response.json()[1].values():
  #     if nid_to_str(node_2) in [i['in_id'], i['out_id']]:
  #       # print(dict({'rel':relation2id_api[i['name']],'weight':i['weight']}))
  #       re[len(re)]= dict({'rel':relation2id_api[i['name']],'weight':i['weight']})
  #   return re


def subgraph(node_ids:list):
  '''
  Get the subgrah by given list of node id

  Args:
    node_ids: list of int , e.g.: [1,2,3,4,5]


  Return:
    subgraph: dict,  same with the API subgraph/query_by_node_name
  '''

  n_list = [nid_to_str(i) for i in node_ids]
  #print(n_list)
  try:
    response = requests.post(
      f"{BASE_URL}/datastores/kg/{KG_NAME}/subgraph/query_by_node_id",
      headers={
        "Authorization": f"Bearer {AUTH_SEC}"
      },
      json={
        "nids": n_list,
        "hops": HOPS
      }
    )
    return response.json()
  except (requests.exceptions.JSONDecodeError, json.decoder.JSONDecodeError):
    response = requests.post(
      f"{BASE_URL}/datastores/kg/{KG_NAME}/subgraph/query_by_node_id",
      headers={
        "Authorization": f"Bearer {AUTH_SEC}"
      },
      json={
        "nids": n_list,
        "hops": HOPS
      }
    )
    print(response.status_code)
    print(n_list)

def subgraph_nodes(node_ids:list):
  # n_list = [nid_to_str(i) for i in node_ids]


  response = requests.request("GET",
                              f"{BASE_URL}/datastores/kg/{KG_NAME}/edges/query_by_id_as_nodes",
                              headers={"Authorization": f"Bearer {AUTH_SEC}"},
                              data=json.dumps(node_ids) )
  return response.json()

def get_subgraph( node_ids ):
  n_list = [nid_to_str(i) for i in node_ids]

  response = requests.request("GET",
                              f"{BASE_URL}/datastores/kg/{KG_NAME}/edges/query_by_id_as_nodes",
                              headers={"Authorization": f"Bearer {AUTH_SEC}"},
                              data=json.dumps(n_list))
  return response.json()

def get_subgraph_by_node_pairs( node_id_pairs ):
  # node_id_pairs = [
  #   [
  #     "n2411",
  #     "n28866"
  #   ],
  #   [
  #     "n13634",
  #     "n134"
  #   ],
  #   [
  #     "n6288",
  #     "n13505"
  #   ]
  # ]
  for node_pair in range(len(node_id_pairs)):
    for i in range(len(node_id_pairs[node_pair])):
      node_id_pairs[node_pair][i] = nid_to_str(node_id_pairs[node_pair][i])
  response = requests.request("GET",
                              f"{BASE_URL}/datastores/kg/{KG_NAME}/nodes/query_nodes_inbetween",
                              headers={"Authorization": f"Bearer {AUTH_SEC}"},
                              data=json.dumps(node_id_pairs))
  return response.json()
def get_edges_by_node_pairs( node_id_pairs ):
  # node_id_pairs = [
  #   [
  #     "n2411",
  #     "n28866"
  #   ],
  #   [
  #     "n13634",
  #     "n134"
  #   ],
  #   [
  #     "n6288",
  #     "n13505"
  #   ]
  # ]

  list_copy = node_id_pairs.copy()
  for node_pair in range(len(list_copy)):
    if list_copy[node_pair][0] ==list_copy[node_pair][1]:
      node_id_pairs.remove(list_copy[node_pair])

      # print(node_id_pairs[node_pair])
  for node_pair in range(len(node_id_pairs)):
    for i in range(len(node_id_pairs[node_pair])):
      node_id_pairs[node_pair][i] = nid_to_str(node_id_pairs[node_pair][i])
  # print(node_id_pairs)
  response = requests.request("GET",
                              f"{BASE_URL}/datastores/kg/{KG_NAME}/edges/query_by_ids",
                              headers={"Authorization": f"Bearer {AUTH_SEC}"},
                              data=json.dumps(node_id_pairs))
  re=dict()
  for i in response.json():
    if i!= dict():
      # print(list(i.keys()))
      # print((i.values()))
      re[list(i.keys())[0]]= dict()
      re[list(i.keys())[0]] = list( i.values())[0]
  # re = response.json()
  # re.remove(dict())
  return re

def main():
 pair_list =  [["n2411","n28866"],[ "n6288","n13505"]]
 pair_list = [[28866, 2411], [6288, 13505],[569,581]]

 qa_nodes = range(250)
 pair_list=[list(i) for i in list(product(qa_nodes, qa_nodes))]
 # print(pair_list)
 pprint.pprint(get_edges_by_node_pairs( pair_list ))
 pass

if __name__ == '__main__':
    main()
