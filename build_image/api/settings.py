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
        "<br/>The API consists of three pre-trained PyTorch models served using TorchServe. "
        "The models are designed to predict the health of crops based on images of the crops. "
        "The models were trained on the following crop types: maize, beans, cocoa, cassava, and banana."
        "<br/>The data were collected from the <a href='https://dataverse.harvard.edu'>Harvard Dataverse</a> and are licensed under the "
        "<a href='https://creativecommons.org/publicdomain/zero/1.0/'>Creative Commons 1.0 DEED license.</a>"
        "<br/>The models differ in the number of classes they predict. The models are:"
        "<br/>1. Binary model: This is a binary model that predicts the health of crops into three classes: healthy and diseased."
        "<br/>2. Single-HLT model: This is a multiclass model that predicts the health of crops into a single healthy (HLT) class and several diseases."
        "<br/>3. Multi-HLT model: This is a multiclass model that predicts the health of crops into multiple healthy (HLT) classes and several diseases."
        "<br/>The key difference between the single-HLT and multi-HLT models is that only the multi-HLT model has a healthy class for each crop type."
        "<br/>The nine specific datasets used can be found at the following URLs:"
        "<br/>1. <a href='https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/R0KL7R'>Spectrometry Cassava Dataset</a>"
        "<br/>2. <a href='https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/T4RB0B'>Cassava Dataset Uganda</a>"
        "<br/>3. <a href='https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GDON8Q'>Maize Dataset Tanzania</a>"
        "<br/>4. <a href='https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6R78HR'>Maize Dataset Namibia</a>"
        "<br/>5. <a href='https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LPGHKK'>Maize Dataset Uganda</a>"
        "<br/>6. <a href='https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TCKVEW'>Beans Dataset Uganda</a>"
        "<br/>7. <a href='https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LQUWXW'>Bananas Dataset Tanzania</a>"
        "<br/>8. <a href='https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BBGQSP'>KaraAgro AI Cocoa Dataset</a>"
        "<br/>9. <a href='https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CXUMDS'>KaraAgro AI Maize Dataset</a>"
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
