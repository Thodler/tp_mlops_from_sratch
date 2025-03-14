import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn

from datetime import datetime
from pydantic import BaseModel

from utils.config_loader import load_config

config = load_config()

app = FastAPI()

class TripRequest(BaseModel):
    pickup_datetime: datetime
    # dropoff_datetime: datetime
    # passenger_count: int
    # pickup_longitude: float
    # pickup_latitude: float
    # dropoff_longitude: float
    # dropoff_latitude: float


@app.get("/")
def root():
    return {"message": "Hello!"}

@app.post("/predict")
def predict(data: TripRequest):

    taxi_model = joblib.load(config['path']['model']+"/"+config['value']['model_name'])

    '''
    {
        "pickup_datetime": "2016-06-01T12:00:00"
    }
    '''

    trip_request = pd.DataFrame([data.model_dump()])

    predict = taxi_model.predict(trip_request)
    return {"predict": predict[0]}


if __name__ == '__main__':
    uvicorn.run("src.api.taxi_controller:app", host="0.0.0.0",
                port=8888, reload=True)