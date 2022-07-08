'''
This file is for using API in QAGNN model.

'''
import requests
import json
import base64
import numpy as np
from io import BytesIO
import time

KG_NAME = "conceptnet"
BASE_URL = 'http://172.29.101.165:7000'
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
  auth_url = "https://square.ukp.informatik.tu-darmstadt.de/auth/realms/square/protocol/openid-connect/token"
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
  auth_sec = auth()
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
    # print(response.status_code)
    # print(n_list)


def main():

 # print(concept2id_api('eating salt'))
 #  print(cpnet_simple_api(20))
 # for i in range(100000)
 # print(id2concept_api(20))
 # for i in range(100000):
 #   print(i)
 #   print(id2concept_api(i))
 #   print(' '.join(id2concept_api(i).split('_')))
 # for i in range(21,10000000):
 #   print(i)
 # print(cpnet_simple_api(20))
 # print(cpnet_has_edge(4,11))
 # print(len(merged_relations()))
 # print(id2relation_api)
 # print(relation2id_api)

 # print(cpnet_values(1271,16783))
 # print(cpnet_has_edge(1271, 1271))
 # print(cpnet_values(1271,1271))
 # start= time.time()
 # print(cpnet_simple_api(551))
 #
 # end = time.time()
 #
 # print(end-start)
 #
 #
 # start= time.time()
 # print(cpnet_simple_api(551))
 #
 # end = time.time()
 #
 # print(end-start)

 #
 # start= time.time()
 #
 # l = [1,2,3,4,5]
 # for i in l:
 #   for t in l:
 #     x= i+t
 #     print(i,t)
 #
 # end = time.time()
 #
 # print(end-start)

 # start = time.time()
 #
 # l = [1, 2, 3, 4, 5]
 # l2 = np.log2(l)
 # # print(l2)
 #
 # end = time.time()
 #
 # print(end - start)
 #
 # start = time.time()
 #
 # l = [1, 2, 3, 4, 5]
 # l2 =[]
 # for i in l:
 #   l2.append(np.log2(i))
 #
 # end = time.time()
 #
 # print(end - start)
 # e = set()
 # e |= set([1,2,3]) &set([2,3,4])
 #
 # print(e)
 # e |= set([7,8,9]) &set([8,9,10])
 # e |= set([1,2,3]) &set([2,3,4])
 # print(e)

 # qa_nodes = set(range(10))
 # start = time.time()
 #
 # extra_nodes = set()
 # for qid in qa_nodes:
 #   for aid in qa_nodes:
 #     # use APi to get extra nodes(?)
 #     csq = cpnet_simple_api(qid)
 #     csa = cpnet_simple_api(aid)
 #     if qid != aid and (csq != False) and (csa != False):
 #
 #       extra_nodes |= set(csq) & set(csa)  # list of node id
 # end = time.time()
 #
 # print(end - start)
 #
 # print(cpnet_has_edge(10, 20))
 # print(list(cpnet_values(10,20).values())==[])
 #
 # start = time.time()
 # node_ids = list(range(10))
 # cids = np.array(node_ids, dtype=np.int32)
 # n_rel = len(id2relation_api)
 # n_node = cids.shape[0]
 # adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
 # for s in range(n_node):
 #   for t in range(n_node):
 #     s_c, t_c = cids[s], cids[t]
 #     # use API to get whethere there is edge (?)
 #
 #     v = cpnet_values(s_c, t_c).values()
 #     if list(v) != []:
 #       for e_attr in v:
 #         if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
 #           adj[e_attr['rel']][s][t] = 1
 #     # if cpnet_has_edge(s_c, t_c):
 #     #     # v = cpnet_values(s_c,t_c)
 #     #     for e_attr in cpnet_values(s_c,t_c).values():
 #     #         if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
 #     #             adj[e_attr['rel']][s][t] = 1
 # end = time.time()
 #
 # print(end - start)
 #
 #
 #
 #
 #
 # start = time.time()
 # node_ids = list(range(10))
 # cids = np.array(node_ids, dtype=np.int32)
 # n_rel = len(id2relation_api)
 # n_node = cids.shape[0]
 # adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
 # for s in range(n_node):
 #   for t in range(n_node):
 #     s_c, t_c = cids[s], cids[t]
 #     # use API to get whethere there is edge (?)
 #
 #
 #     if cpnet_has_edge(s_c, t_c):
 #         # v = cpnet_values(s_c,t_c)
 #         for e_attr in cpnet_values(s_c,t_c).values():
 #             if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
 #                 adj[e_attr['rel']][s][t] = 1
 # end = time.time()
 # print()
 # print(end - start)
 # print(cpnet_values(1271,16783))

 # qa_nodes = np.array([1,2,3,4,5])
 # idx = np.where(qa_nodes==5)[0][0]
 # print(idx)
 # # print(set([nid_to_int(i) for i in list(subgraph(set(qa_nodes))[0].keys())]))
 # print(subgraph(set(qa_nodes))[0].values())
 # n_rel = len(id2relation_api)
 # adj = np.zeros((n_rel, 5, 5), dtype=np.uint8)
 # for i in subgraph(qa_nodes)[1].values():
 #   # print(i)
 #   # print(i['in_id'])
 #   # print(i['name'])
 #   # print(i['weight'])
 #   # print(relation2id_api[i['name']])
 #
 #   if ( nid_to_int(i['in_id']) in qa_nodes and nid_to_int(i['out_id']) in qa_nodes):
 #     if relation2id_api[i['name']] >= 0 and relation2id_api[i['name']] < n_rel:
 #      print(i)
 #      in_idx = np.where(qa_nodes==nid_to_int(i['in_id']))[0][0]
 #      out_idx = np.where(qa_nodes == nid_to_int(i['out_id']))[0][0]
 #      adj[relation2id_api[i['name']]][in_idx ][ out_idx ]=1
 # # print(qa_nodes.index(3))
 nodes = ['n29187', 'n19977', 'n4621', 'n155667', 'n22553', 'n6699', 'n79923', 'n15925', 'n6198', 'n18487', 'n20023', 'n121398', 'n217658', 'n49718', 'n579', 'n7235', 'n6729', 'n4170', 'n9810', 'n264789', 'n183897', 'n6234', 'n21085', 'n3678', 'n14439', 'n15987', 'n4212', 'n24693', 'n7799', 'n171133', 'n20095', 'n6787', 'n17031', 'n654', 'n21649', 'n16534', 'n2711', 'n76446', 'n17055', 'n73892', 'n12452', 'n12454', 'n74928', 'n14512', 'n156850', 'n692', 'n1718', 'n695', 'n563382', 'n191166', 'n17088', 'n28866', 'n28868', 'n1736', 'n20177', 'n80102', 'n746', 'n1770', 'n749', 'n156909', 'n751', 'n22256', 'n239', 'n15092', 'n2296', 'n410362', 'n3326', 'n108801', 'n5895', 'n94984', 'n20744', 'n5898', 'n4363', 'n113420', 'n6920', 'n524046', 'n102159', 'n4364', 'n13578', 'n102163', 'n4374', 'n173855', 'n19251', 'n15674', 'n89933', 'n12112', 'n2389', 'n5975', 'n401240', 'n23384', 'n74590', 'n1800', 'n1380', 'n159589', 'n1381', 'n18282', 'n2411', 'n2412', 'n2413', 'n1386', 'n2415', 'n19819', 'n1911', 'n3454', 'n11137', 'n2434', 'n8067', 'n2436', 'n3461', 'n8069', 'n18311', 'n2438', 'n208789', 'n12694', 'n22942', 'n16286', 'n928', 'n26016', 'n26018', 'n931', 'n11171', 'n5539', 'n934', 'n26022', 'n9138', 'n363959', 'n28089', 'n74683', 'n270272', 'n2496', 'n4544', 'n8131', 'n3013', 'n439239', 'n1993', 'n7116', 'n3030', 'n3035', 'n3036', 'n271326', 'n2527', 'n72676', 'n23013', 'n21492']

 qa_nodes = [nid_to_int(i)  for i in nodes]
 print(len(nodes))
 print(subgraph(set(qa_nodes)))





 pass

if __name__ == '__main__':
    main()
