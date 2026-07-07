from pydantic import BaseModel

class Transaction(BaseModel):
    features: list[float]