from pydantic import BaseModel


class ModelIdentifier(BaseModel):
    language: str
    model_name: str
    is_stemmed: bool = False

    def __hash__(self):
        return hash(f"{self.language}-{self.model_name}-{self.is_stemmed}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelIdentifier):
            return False
        return (
            self.language == other.language
            and self.model_name == other.model_name
            and self.is_stemmed == other.is_stemmed
        )

    def __str__(self) -> str:
        return f"{self.language}/{self.model_name}"
