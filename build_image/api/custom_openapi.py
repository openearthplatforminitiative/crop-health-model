import logging
import os
from pathlib import Path
from string import Template

from api.settings import settings

supported_languages = {"cURL": "sh", "JavaScript": "js", "Python": "py"}


def custom_openapi_gen(openapi_schema: dict, example_code_dir: Path):
    openapi_schema["info"]["x-logo"] = {
        "url": f"https://{settings.api_domain}/assets/icons/open-epi-logo.svg"
    }
    openapi_schema["info"]["title"] = settings.title
    openapi_schema["info"]["version"] = settings.version
    openapi_schema["info"]["description"] = settings.api_description

    # Endpoint path and method to modify
    endpoint_path = "/predictions/{model_name}"
    method = "post"

    # Need to manually define the schema for the model response
    if (
        endpoint_path in openapi_schema["paths"]
        and method in openapi_schema["paths"][endpoint_path]
    ):
        openapi_schema["paths"][endpoint_path][method]["responses"]["200"] = {
            "description": (
                "Predicted class confidences, all summing to 1.0.",
                " Actual class names may vary by model.",
            ),
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["class1", "class2", "class3"],
                        "properties": {
                            "class1": {
                                "type": "float",
                                "description": "Confidence score for class1.",
                            },
                            "class2": {
                                "type": "float",
                                "description": "Confidence score for class2.",
                            },
                            "class3": {
                                "type": "float",
                                "description": "Confidence score for class3.",
                            },
                        },
                        "example": {"class1": 0.85, "class2": 0.10, "class3": 0.05},
                    }
                }
            },
        }

    # # Derive the API routes from the OpenAPI schema
    # api_routes = list(openapi_schema["paths"].keys())
    # # remove leading slashes from the routes
    # api_routes = [route.lstrip("/") for route in api_routes]

    # for route in api_routes:
    #     code_samples = get_code_samples(route, example_code_dir)
    #     if code_samples:
    #         # add leading slashes back to the routes
    #         route = "/" + route
    #         if "get" in openapi_schema["paths"][route]:
    #             openapi_schema["paths"][route]["get"]["x-codeSamples"] = code_samples
    #         elif "post" in openapi_schema["paths"][route]:
    #             openapi_schema["paths"][route]["post"]["x-codeSamples"] = code_samples

    # return openapi_schema


def get_code_samples(route: str, example_code_dir: Path):
    code_samples = []
    normalized_route_name = route.replace("/", "__").replace("{", "").replace("}", "")
    for lang, file_ext in supported_languages.items():
        file_with_code_sample = (
            example_code_dir / lang.lower() / f"{normalized_route_name}.{file_ext}"
        )
        print(file_with_code_sample)
        if os.path.isfile(file_with_code_sample):
            with open(file_with_code_sample) as f:
                code_template = Template(f.read())
                code_samples.append(
                    {
                        "lang": lang,
                        "source": code_template.safe_substitute(
                            api_url=settings.api_url,
                        ),
                    }
                )
        else:
            logging.warning(
                "No code sample found for route %s and language %s",
                route,
                lang,
            )
    return code_samples
