from pydantic import BaseModel

class PredictionResponse(BaseModel):
    city: str
    prediction_24h: float
    prediction_48h: float
    prediction_72h: float