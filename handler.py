import pickle 
import pandas as pd 
from flask import Flask,request,Response 
from healthinsurance.HealthInsurance import HealthInsurance
import os 

# loading Model
model = pickle.load(open("model/model_logistic_regression.pkl","rb"))

app = Flask(__name__)
@app.route("/predict",methods=["POST"])
def health_Insurance():
    test_json  = request.get_json()
    if test_json: # se existir dados
        if isinstance(test_json,dict): # se for uma instancia de dicionario (json)
            test_raw = pd.DataFrame(test_json, index=[0]) # unique example
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys()) # multiple examples           
         # Instantiate HealthInseurance  class
        pipeline = HealthInsurance()
         # data cleaning
        df1 = pipeline.data_cleaning(test_raw) 
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        # data preparation
        df3 = pipeline.data_preparation(df2)
        # prediction
        df_response = pipeline.get_prediction(model,test_raw,df3)
        return df_response
    else:
        return Response("{}",status=200,mimetype="application/json")
if __name__=="__main__": 
    port= int(os.environ.get('PORT',5000))   
    app.run(host='0.0.0.0',port=port)
     