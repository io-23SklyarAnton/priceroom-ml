import numpy as np
from fastapi_utilities import repeat_every

from fastapi import APIRouter, HTTPException

from model_handler import ModelHandler

from schemas.realty import RealtyPredictionBody, RealtyPredictionResponse

router = APIRouter()

predictions = {}
model_handler = ModelHandler()
alarmed_regions = {}


async def load_model():
    model_handler.load_model()
    print("Model loaded.")


@repeat_every(seconds=60 * 60)
async def regular_train_model():
    model_handler.train_model()
    model_handler.load_model()


@router.post("/api/predict-price/")
def predict_price(realty: RealtyPredictionBody):
    try:
        model_inputs = model_handler.prepare_features(
            district=realty.district,
            rooms_count=realty.rooms_count,
            total_square_meters=realty.total_square_meters,
        )

        y_pred = model_handler.model.predict(model_inputs)

        if not np.isfinite(y_pred[0]):
            raise ValueError("Model prediction returned a non-finite value (NaN or Infinity).")

        return RealtyPredictionResponse(price=float(y_pred[0]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
