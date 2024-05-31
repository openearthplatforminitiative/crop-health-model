import logging
import os
from pathlib import Path
from string import Template

from fastapi.routing import APIRoute

from api.settings import settings

supported_languages = {"cURL": "sh", "JavaScript": "js", "Python": "py"}


def custom_openapi_gen(openapi_schema: dict, example_code_dir: Path):

    openapi_schema["info"]["x-logo"] = {
        "url": f"https://{settings.api_domain}/assets/icons/open-epi-logo.svg"
    }
    openapi_schema["title"] = settings.title
    openapi_schema["version"] = settings.version
    openapi_schema["description"] = settings.api_description

    # Derive the API routes from the OpenAPI schema
    api_routes = list(openapi_schema["paths"].keys())

    # remove leading slashes from the routes
    api_routes = [route.lstrip("/") for route in api_routes]

    for route in api_routes:
        code_samples = get_code_samples(route, example_code_dir)

        if code_samples:
            # add leading slashes back to the routes
            route = "/" + route
            # try first with "get", then with "post"
            if "get" in openapi_schema["paths"][route]:
                openapi_schema["paths"][route]["get"]["x-codeSamples"] = code_samples
            elif "post" in openapi_schema["paths"][route]:
                openapi_schema["paths"][route]["post"]["x-codeSamples"] = code_samples

    return openapi_schema


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
