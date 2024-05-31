from httpx import Client

with Client() as client:
    # Get the prediction for image cocoa.jpg passed as a binary file in the request body
    response_binary = client.post(
        url="$api_url" + "/predictions/binary",
        body=open("cocoa.jpg", "rb").read(), 
    )

    data_binary = response_binary.json()
    print(data_binary["HLT"])

    # Get the prediction for image cocoa.jpg passed as a binary file in the request body
    response_single_HLT = client.post(
        url="$api_url" + "/predictions/single-HLT",
        body=open("cocoa.jpg", "rb").read(), 
    )

    data_single_HLT = response_single_HLT.json()
    print(data_single_HLT)

    # Get the prediction for image cocoa.jpg passed as a binary file in the request body
    response_multi_HLT = client.post(
        url="$api_url" + "/predictions/multi-HLT",
        body=open("cocoa.jpg", "rb").read(), 
    )

    data_multi_HLT = response_multi_HLT.json()
    print(data_multi_HLT)
