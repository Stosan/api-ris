from typing import *
import base64
import json
import os
from fastapi import *

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.classifiers.color_classifier.color_classifer import ColorClassifier

app = FastAPI()

app.mount("/app/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/")
def read_root():
    return {   "status" : "200",
                "response" : "Color Classification Api. Built for Risigner Project",
                "Author":"Sam Ayo"}


@app.get("/who")
def read_auth():
    return {   "status" : "200",
                "response" : "Hi Ria, I built this because I love you!"}


@app.get("/colors_classify/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("viewer.html", {"request": request})


@app.post("/color_predict/")
async def Predict_color(file: bytes = File(...)):
    try:
        #base64_to_image(file)
        classifier = ColorClassifier()
        predictions = classifier.compute_prediction(file)
        if predictions is not None:
                spc_color = predictions["spc_color"] if "spc_color" in predictions else "error"
                return_data = predictions
        else :
            return_data = {"error" : "4", "message" : f"Error : "}
        
    except Exception as e:
        return_data = {
        "error" : "3",
        "message" : f"Error : {str(e)}",
       
        }
        
    return return_data