import json
import requests
# import ipdb

base_url = "http://localhost:7000"

def get_token(print_json=False):
  response = requests.post(
      "https://square.ukp.informatik.tu-darmstadt.de/auth/realms/square/protocol/openid-connect/token",
      data={
          "grant_type": "password",
          "client_id": "web-app",
          "username": "haoz",
          "password": "VQwB9EPF8qYpeux"
      }
  )
  response_json = response.json()
  if print_json:
    print(json.dumps(response_json, indent=4))
  return response_json["access_token"]

if __name__ == "__main__":
  response = requests.get(
    f"{base_url}/datastores", 
    headers={
        "Authorization": f"Bearer {get_token()}"
    }
  )
  print(json.dumps(response.json(), indent=4))

  # ipdb.set_trace()



  response = requests.get(
    f"{base_url}/datastores/conceptnet/stats", 
    headers={
        "Authorization": f"Bearer {get_token()}"
    }
  )
  print(json.dumps(response.json(), indent=4))