import joblib
from fastapi import FastAPI, HTTPException
import uvicorn

from datetime import datetime
from pydantic import BaseModel

from utils.config_loader import load_config

config = load_config()

app = FastAPI()

class TripRequest(BaseModel):
    id: str
    vendor_id: int
    pickup_datetime: datetime
    dropoff_datetime: datetime
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str
    trip_duration: int


@app.get("/")
def root():
    return {"message": "Hello!"}

@app.post("/predict")
async def predict():

    model = joblib.load(config['path']['model'], config['value']['model_name'])

    return {"item_id": "item"}


if __name__ == '__main__':
    uvicorn.run("src.api.taxi_controller:app", host="0.0.0.0",
                port=8888, reload=True)