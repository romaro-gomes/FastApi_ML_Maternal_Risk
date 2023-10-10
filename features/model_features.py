from pydantic import BaseModel
class MaternalFeatures(BaseModel):
    Age:int
    SystolicBP:float
    DiastolicBP:float
    BS:float
    BodyTemp:float
    HeartRate:float