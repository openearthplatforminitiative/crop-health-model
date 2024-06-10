import pathlib
from contextlib import asynccontextmanager

import requests
from api.custom_openapi import custom_openapi_gen
from api.settings import settings
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

example_code_dir = pathlib.Path(__file__).parent / "example_code"
openapi_json_cache = None


@asynccontextmanager
async def app_lifespan(app):
    global openapi_json_cache
    try:
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
        else:
            raise Exception("Failed to load OpenAPI JSON from TorchServe")
        yield
    except Exception as e:
        print(f"An error occurred: {e}")


app = FastAPI(openapi_url=None, lifespan=app_lifespan)


@app.get("/torchserve-openapi.json")
async def get_openapi_json():
    if openapi_json_cache:
        return openapi_json_cache
    else:
        raise HTTPException(status_code=500, detail="OpenAPI JSON is not loaded")


@app.get("/redoc", response_class=HTMLResponse)
async def redoc():
    base_url = f"http://127.0.0.1:{settings.uvicorn_port}"
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
        <redoc spec-url='{base_url}/torchserve-openapi.json'></redoc>
        <script src="https://cdn.jsdelivr.net/npm/redoc/bundles/redoc.standalone.js"></script>
    </body>
    </html>
    """
    return redoc_html


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.uvicorn_host, port=settings.uvicorn_port)
