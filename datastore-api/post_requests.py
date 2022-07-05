####     Only some requests for testing
kg_name = "conceptnet"
import requests
import json
from get_start.easy_conn import get_token, base_url
base_url = 'http://172.29.101.165:7000'
# Check if Kg exists
response = requests.get(
f"{base_url}/datastores/kg", 
headers={
    "Authorization": f"Bearer {get_token()}"
}
)


# Get subgraph
# response = requests.post(
#     f"{base_url}/datastores/kg/{kg_name}/subgraph/query_by_node_name", 
#     headers={
#         "Authorization": f"Bearer {get_token()}"
#     },
#    json={
#           "nids": ['obama', 'united_states_of_america', 'japan'],
#           "hops": 2 
#           }
# )
# response -> nodes ,edges

# Get all relations
# response = requests.get(
# f"{base_url}/datastores/kg/{kg_name}/relations", 
# headers={
#     "Authorization": f"Bearer {get_token()}"
# }
# )


# ##  Get node by id
# node_id = "n3"
# response = requests.get(
#     f"{base_url}/datastores/kg/{kg_name}/{node_id}", 
#     headers={
#         "Authorization": f"Bearer {get_token()}"
#     }
# )



# ## Get node by the name
# response = requests.get(
#     f"{base_url}/datastores/kg/{kg_name}/nodes/query_by_name",
#     headers={
#         "Authorization": f"Bearer {get_token()}"
#     },
#     data={
#         'doc_id': 'absoluteness'
#         }
# )


# ## Update node by id into somethin else
# node_id = "n41714"
# response1 = requests.put(
#     f"{base_url}/datastores/kg/{kg_name}/node/{node_id}", 
#     headers={
#         "Authorization": f"Bearer {get_token()}"
#     },
#    json= 
#           {
#             '_id': 'n41714',
#             'description': 'me',
#             'in_id': None,
#             'name': 'me',
#             'out_id': None,
#             'type': 'node',
#             'weight': None
#           }
# )


# ##  Does not work yet
# from typing import Tuple
# #  Get edge for a given node pair
# response = requests.get(
#     f"{base_url}/datastores/kg/{kg_name}/edges/query_by_name",
#     headers={
#         "Authorization": f"Bearer {get_token()}"
#     },
#     data={
#         'doc_id': ['n32369','n32370']
#         }
# )


# ##  Get node by id
# node_id = "n3"
# node_name= "actinal"
# response = requests.get(
#     f"{base_url}/datastores/kg/{kg_name}/{node_id}", 
#     headers={
#         "Authorization": f"Bearer {get_token()}"
#     }
# )



response = requests.post(
    f"{base_url}/datastores/kg/{kg_name}/nodes/query_by_name",
    headers={
        "Authorization": f"Bearer {get_token()}"
    },
    data= json.dumps(['actinal']) 
)

# node_id = "n3"
# response = requests.get(
#     f"{base_url}/datastores/kg/{kg_name}/{node_id}", 
#     headers={
#         "Authorization": f"Bearer {get_token()}"
#     }
# )


# response = requests.post(
#     f"{base_url}/datastores/kg/{kg_name}/subgraph/query_by_node_name", 
#     headers={
#         "Authorization": f"Bearer {get_token()}"
#     },
#    json={
#           "nids": ['apple'],
#           "hops": 2
#           }
# )

def nid_to_int(node_id:str):
    '''
    Convert node id for string to int
        'n12345'--> 12345

    Args:
        node_id: str , e.g. 
    
    Return:
        node id : int

    '''
    node_id= node_id.replace('n','')
    #print(node_id)
    
    return int(node_id)

def nid_to_str(node_id:int):
    '''
    Convert node id for int to string
    12345-->'n12345'
   

    Args:
        node_id: int
    
    Return:
        node id : str

    '''
    node_id= 'n'+str(node_id)
    print(node_id)
    
    return node_id

response = requests.post(
    f"{base_url}/datastores/kg/{kg_name}/subgraph/query_by_node_id", 
    headers={
        "Authorization": f"Bearer {get_token()}"
    },
   json={
          "nids": ['n17363','n3195'],
          "hops": 2
          }
)



import pprint
pp = pprint.PrettyPrinter(indent=4)
stuff = response.json()
# for i in stuff:
#     # print(i)
#     if 'n16783' ==i['out_id']:
#         print(dict({'rel':i['name'],'weight':i['weight']}))
#         # print(i['weight'])
#         # print(i['name'])
print(response.status_code)
pp.pprint(stuff)
# print(response.text)


# import numpy as np
# print([ f'n{np.random.randint(1,3287,size=1)[0]}' for i in range(np.random.randint(100,150,size=1)[0])])
# print([ np.random.randint(1,3287,size=1)[0] for i in range(np.random.randint(100,150,size=1)[0])])

