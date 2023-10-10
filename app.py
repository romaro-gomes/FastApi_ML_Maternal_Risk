from fastapi import FastAPI, Request, HTTPException,Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import pickle

import pandas as pd

from features.model_features import MaternalFeatures

app=FastAPI()

templates = Jinja2Templates(directory="templates")

model=pickle.load(open('model.pkl','rb'))
preprocessor=pickle.load(open('preprocessor.pkl','rb'))


def get_values(data):
    return [getattr(data,field) for field in data.__annotations__.keys()]

@app.get('/',response_class=HTMLResponse)
async def home(request: Request):
     return templates.TemplateResponse('home.html', {'request':request})

@app.post('/predict')
async def predict( request: Request,
                    age:int  = Form(...),
                    systolic_bp: float = Form(...),
                    diastolic_bp: float = Form(...),
                    bs: float = Form(...),
                    body_temp: float = Form(...),
                    heart_rate: float = Form(...)):

    if (age < 18 or age > 99) or\
        (systolic_bp < 60 or systolic_bp >220) or\
        (diastolic_bp <45 or diastolic_bp > 110) or\
        (bs<6 or bs>20) or\
        (body_temp<95 or body_temp>110)or\
        (heart_rate<7 or heart_rate>100):
        raise HTTPException(status_code=400, detail="Some value are out of range")
    
    
    data = MaternalFeatures(
        Age=age,
        SystolicBP=systolic_bp,
        DiastolicBP=diastolic_bp,
        BS=bs,
        BodyTemp=body_temp,
        HeartRate=heart_rate
    )

    data_df = pd.DataFrame([get_values(data)], columns=data.__annotations__.keys())
    data_processed=preprocessor.transform(data_df)
    prediction = model.predict(data_processed)
    proba=model.predict_proba(data_processed)
    probability=f"{((proba.tolist()[0][1])*100):.2f}% {((proba.tolist()[0][2])*100):.2f}% {((proba.tolist()[0][0])*100):.2f}%"

    if prediction[0] == 0:
        pred='High Risk'
    elif prediction[1]:
        pred='Low Risk'
    else:
        pred='Medium Risk'

    return templates.TemplateResponse("predict.html", {"request": request, "prediction": pred, "probability":probability} )
                                      