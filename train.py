# we load the Ml model here and predict the output:

import numpy as np
import pandas as pd
import joblib 
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("hiring.csv")
print(dataset)

dataset.experience.fillna(0,inplace =True)

dataset.test_score.fillna(dataset.test_score.mean(),inplace =True)

X= dataset.iloc[:,:3]

def convert_to_int(word):
    word_dict = {
        "one":1,
        "two":2,
        "three":3,
        "four":4,
        "five":5,
        "six":6,
        "seven":7,
        "eight":8,
        "nine":9,
        "ten":10,
        "eleven":11,
        0:0
                }

    return word_dict[word]

X["experience"] = X.experience.apply(convert_to_int)
# print(X["experience"])
y = dataset.iloc[:,-1]

reg = LinearRegression()
reg.fit(X,y)

print("Model training is done")

# save the model:
joblib.dump(reg,"hiring_model.pkl")

print(reg.predict([[1,8,9]]))

 