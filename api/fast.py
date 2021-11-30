from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
import joblib
from deep_music.predict import predict_basic
from fastapi import FastAPI, File, UploadFile
from deep_music.magenta_testing import PreTrainedRnn
from fastapi.responses import StreamingResponse, FileResponse
import json

MODEL_PATH = 'deep_music/data/magenta_models/basic_rnn.mag'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict_basic")
def basic_predict(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20):
    # list_notes = np.array(list_notes)
    # return predict_basic(list_notes, 20, [], 20)
    string_input = [
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16,
        a17, a18, a19, a20
    ]
    number_input = []
    for i in string_input:
        number_input.append(int(i))
    X = np.array(number_input).reshape(20, 1)
    prediction = predict_basic(X, a18, [], 20)
    return prediction


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    model = PreTrainedRnn(MODEL_PATH)
    # json.loads(file)
    contents = file.file
    contents = await file.read()

    with open(file.filename, "wb") as f:
        f.write(contents)



    # print(file)
    melody = model.melody_to_seq(file.filename)
    model.generate_new_melody(500, 1, melody, "test_midi.mid")

    return FileResponse("deep_music/data/results/test_midi.mid", media_type="audio/mid")
