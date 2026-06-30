import pandas as pd 
import joblib 
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import List , Annotated , Literal
from pydantic import BaseModel , Field


#load the saved model
model = joblib.load("models/xgb_insurance.pkl")
app = FastAPI(title="Medical Insurance Cost Predictor")

#Pydantic model (For validating the input data)
class InsuranceInput(BaseModel):
    age      : int   = Field(..., gt=0, lt=120, description="Age of the member", examples=[35])
    bmi      : float = Field(..., gt=0, description="BMI of the member", examples=[28.5])
    children : int   = Field(..., ge=0, description="Number of children", examples=[2])
    sex      : Literal["male", "female"] = Field(..., description="Gender of the member")
    smoker   : Literal["yes", "no"] = Field(..., description="Smoker Yes/No")
    region   : Literal["northeast", "northwest", "southeast", "southwest"] = Field(..., examples=["southeast"])


def preprocess(data : InsuranceInput):
    df = pd.DataFrame({
        "age"               : [data.age],
        "bmi"               : [data.bmi], 
        "children"          : [data.children],
        "sex_male"          : [1.0 if data.sex == "male" else 0.0],
        "smoker_yes"        : [1.0 if data.smoker == "yes" else 0.0],
        "region_northwest"  : [1.0 if data.region == "northwest" else 0.0],
        "region_southeast"  : [1.0 if data.region == "southeast" else 0.0],
        "region_southwest"  : [1.0 if data.region == "southwest" else 0.0],
    })
    return df.astype(float)

@app.get("/about")
def about():
    return {
        "message" : "This app is used to predict the insurance charges of the member."
    }

@app.get("/health")
def health():
    return{
        "status" : "OK"
    }

@app.post("/predict")
def predict(data : InsuranceInput):
    input_df = preprocess(data)
    prediction = model.predict(input_df)[0]

    return JSONResponse(
        status_code=200 , 
        content={
            "status" : "success",
            "Predicted Charge" : float(round(float(prediction) , 2)),
            "message" : "Predicted Sucessfully" 
        }
    )
