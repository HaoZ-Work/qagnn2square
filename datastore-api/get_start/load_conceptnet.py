from html import entities
import requests
from easy_conn import get_token, base_url
import json
from tqdm import auto

nodes = {}
nids = {}
edges = []


url = "https://square.ukp.informatik.tu-darmstadt.de/auth/realms/square/protocol/openid-connect/token"

payload='grant_type=password&client_id=web-app&username=haoz&password=VQwB9EPF8qYpeux'
headers = {
  'Content-Type': 'application/x-www-form-urlencoded'
}

response = requests.request("POST", url, headers=headers, data=payload)


sec = response.json()['access_token']

def get_sec():
    url = "https://square.ukp.informatik.tu-darmstadt.de/auth/realms/square/protocol/openid-connect/token"

    payload='grant_type=password&client_id=web-app&username=haoz&password=VQwB9EPF8qYpeux'
    headers = {
    'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.request("POST", url, headers=headers, data=payload)


    sec = response.json()['access_token']
    return sec


with open('data/conceptnet.en.csv', 'r') as f:
    for line in f:
        items = line.strip().split('\t')
        rel, head, tail, weight = items
        for entity_name in [head, tail]:
            if entity_name in nodes:
                continue

            nid = f'n{len(nodes)}'
            node = {
                'id': nid,
                'name': entity_name,
                'type': 'node',
                # 'description': '',
                'description': entity_name.replace('_', ' '),
                'weight': None,
                'in_id': None,
                'out_id': None
            }
            nodes[entity_name] = node
            nids[entity_name] = nid
        
        eid = f'e{len(edges)}'
        edge = {
            'id': eid,
            'name': rel,
            'type': 'edge',
            'description': '',
            'weight': float(weight),
            'in_id': nids[head],
            'out_id': nids[tail]
        }
        edges.append(edge)

datastore_name = 'conceptnet'

for data in [list(nodes.values()), edges]:
    batch_size = 5000
    for b in auto.tqdm(range(0, len(data), batch_size)):
        response = requests.post(
            f'http://127.0.0.1:7000/datastores/kg/{datastore_name}/nodes',
            headers={
                "Authorization": f"Bearer {get_sec()}"
            },
            json=data[b:b+batch_size]
        )
        print(response.json())
        # assert response.json()['code'] == 200
