import pathlib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse 
import requests
from api.custom_openapi import custom_openapi_gen

app = FastAPI(openapi_url=None)

example_code_dir = pathlib.Path(__file__).parent / "example_code"

# Cache the OpenAPI JSON
openapi_json_cache = None

@app.on_event("startup")
async def load_openapi_json():
    global openapi_json_cache
    response = requests.options("http://localhost:8080/", timeout=10)
    # log the response
    if response.status_code == 200:
        openapi_json_cache = response.json()
        print("Response from TorchServe")
        print(openapi_json_cache)
    else:
        raise Exception("Failed to load OpenAPI JSON from TorchServe")
    

@app.on_event("startup")
async def load_openapi_json():
    global openapi_json_cache
    response = requests.options("http://localhost:8080/", timeout=10)
    if response.status_code == 200:
        openapi_json = response.json()
        
        # Remove specific endpoints if needed
        endpoints_to_keep = ["/predictions/{model_name}", "/ping"]
        all_endpoints = list(openapi_json["paths"].keys())
        for endpoint in all_endpoints:
            if endpoint not in endpoints_to_keep:
                del openapi_json["paths"][endpoint]

        openapi_json = custom_openapi_gen(openapi_json, example_code_dir)

        openapi_json_cache = openapi_json
        print("Modified OpenAPI JSON loaded")
    else:
        raise Exception("Failed to load OpenAPI JSON from TorchServe")

@app.get("/torchserve-openapi.json")
async def get_openapi_json():
    if openapi_json_cache:
        return openapi_json_cache
    else:
        raise HTTPException(status_code=500, detail="OpenAPI JSON is not loaded")

@app.get("/redoc", response_class=HTMLResponse)
async def redoc():
    redoc_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ReDoc Documentation</title>
        <!-- Redoc stylesheet -->
        <link href="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.css" rel="stylesheet">
    </head>
    <body>
        <!-- Redoc script that builds the page from OpenAPI spec -->
        <redoc spec-url='/torchserve-openapi.json'></redoc>
        <script src="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js"></script>
    </body>
    </html>
    """
    return redoc_html

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)