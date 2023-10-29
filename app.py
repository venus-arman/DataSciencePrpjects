import glob, os
# import pandas
import speech_recognition as sr
from flask import Flask
from flask_restful import Api, Resource, reqparse
import random
import pickle


app = Flask(__name__)
api = Api(app)

# Opening saved model
with open("voice_txt.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/written_records")
def write_the_record():
    # json_ = reqparse.request.json()
    # query_df = pd.DataFrame(json_)
    path = "C:/Users/syous/OneDrive/Documents/Freelance projects/DataSciencePrpjects/Voice records/"
    text_list = model.voice_to_text(path)# Passing in variables for prediction
    return (text_list)

if __name__ == "__write_the_record__":
    app.run(debug=True)
