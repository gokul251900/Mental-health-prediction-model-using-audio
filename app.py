from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import json

from predict import predict_audio

from database import engine, SessionLocal
from models import Prediction, Base

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    path = "temp.wav"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_audio(path)

    db = SessionLocal()

    record = Prediction(
        main_prediction=result["main_prediction"],
        predicted_disorders=json.dumps(result["predicted_disorders"]),
        probabilities=json.dumps(result["probabilities"])
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    db.close()

    return {"result_id": record.id}


@app.get("/result/{result_id}")
def get_result(result_id: int):

    db = SessionLocal()

    record = db.query(Prediction).filter(Prediction.id == result_id).first()
    db.close()

    if not record:
        return {"error": "Result not found"}

    return {
        "result_id": record.id,
        "main_prediction": record.main_prediction,
        "predicted_disorders": json.loads(record.predicted_disorders),
        "probabilities": json.loads(record.probabilities)
    }