from httpx import Client

with Client() as client:
    # Get the binary model prediction for image cocoa.jpg
    # passed as a binary file in the request body
    response_binary = client.post(
        url="$api_url" + "/predictions/binary",
        data=open("cocoa.jpg", "rb").read(),
    )

    data_binary = response_binary.json()
    # Print the prediction for the healthy class
    print(data_binary["HLT"])

    # Get the single-HLT model prediction for image cocoa.jpg
    # passed as a binary file in the request body
    response_single_HLT = client.post(
        url="$api_url" + "/predictions/single-HLT",
        data=open("cocoa.jpg", "rb").read(),
    )

    data_single_HLT = response_single_HLT.json()
    # Print the top 5 predictions
    print(data_single_HLT)

    # Get the multi-HLT model prediction for image cocoa.jpg
    # passed as a binary file in the request body
    response_multi_HLT = client.post(
        url="$api_url" + "/predictions/multi-HLT",
        data=open("cocoa.jpg", "rb").read(),
    )

    data_multi_HLT = response_multi_HLT.json()
    # Print the top 5 predictions
    print(data_multi_HLT)
