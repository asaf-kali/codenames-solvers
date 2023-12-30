from pydantic import BaseModel


class ModelIdentifier(BaseModel):
    language: str
    model_name: str
    is_stemmed: bool = False

    def __hash__(self):
        return hash(f"{self.language}-{self.model_name}-{self.is_stemmed}")
