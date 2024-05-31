from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    version: str = "0.0.1"
    title: str = "Crop Health API"
    uvicorn_port: int = 5000
    uvicorn_host: str = "0.0.0.0"
    uvicorn_reload: bool = True
    uvicorn_proxy_headers: bool = False
    api_root_path: str = ""
    api_description: str = (
        "This is a RESTful service that provides predictions for crop health."
        "The API consists of three pre-trained PyTorch models served using TorchServe."
        " The models are designed to predict the health of crops based on images of the crops."
        "The models were trained on the following crop types: maize, beans, cocoa, cassava, and banana."
        "The data were collected from the Harvard Dataverse and are licensed under the Creative Commons 1.0 DEED license."
        "The models differ in the number of classes they predict. The models are:"
        "1. Binary model: This is a binary model that predicts the health of crops into three classes: healthy and diseased."
        "2. Single-HLT model: This is a multiclass model that predicts the health of crops into a single healthy (HLT) class and several diseases."
        "3. Multi-HLT model: This is a multiclass model that predicts the health of crops into multiple healthy (HLT) classes and several diseases."
        "The key difference between the single-HLT and multi-HLT models is that the multi-HLT model has a healthy class for each crop type."
        ""
    )
    api_domain: str = "localhost"

    @property
    def api_url(self):
        if self.api_domain == "localhost":
            return f"http://{self.api_domain}:{self.uvicorn_port}"
        else:
            return f"https://{self.api_domain}{self.api_root_path}"


settings = Settings()
