from httpx import Client

with Client() as client:
    response = client.get(url="$api_url" + "/ping")
    data = response.json()
