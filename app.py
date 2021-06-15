from logging import debug
from flask import Flask,render_template, request
import joblib
import numpy as np
from numpy.core.numeric import outer


# create a flask app
app = Flask(__name__)  # __name__ is the name that we have given to this file

# load the model:
model = joblib.load("hiring_model.pkl")

@app.route('/')
# write a function:
def welcome():
    return render_template('base2.html')

@app.route("/predict",methods = ["POST"])

def predict():
    # get the info:
    exp = request.form.get('experience')     
    score = request.form.get('test_score')  
    interview_score = request.form.get('interview_score')
     
    # need to convert the type of data
    prediction = model.predict(([[int(exp),int(score),int(interview_score)]]))
    
    output = round(prediction[0],2)
    # print(prediction)
    # this return is somthing we r returning from flask to html so in html need to use  jinja formate {{}}
    return render_template('base2.html', Prediction_text = f"Employee Salary will be ${output}")


if __name__ == '__main__':
    app.run(debug =True)


 