import pickle
from flask import Flask,request,render_template

import pandas as pd
import numpy as np
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
All_stopwords= stopwords.words("english")
All_stopwords.remove("not")
app = Flask(__name__)

countvectorizer = pickle.load(open("CountVectorizer.pkl","rb"))
gaussian = pickle.load(open("Gaussian.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/stem", methods= ["POST"])

def stem():
    string= request.form["review"]
    ##Now it's time to make it countvectorizable
    
    ps  = PorterStemmer()
    r = re.sub("[^a-zA-Z]"," ", string)
    r = r.lower()
    r= r.split()
    
    for j in r:
        if not j in All_stopwords:
            r= [ps.stem(j)]
    r = " ".join(r)
    r =[r]
    data = countvectorizer.transform(r).toarray()
    output= gaussian.predict(data)
    if output==1:
        answer= "Yes, it's a nice review"
    else:
        answer = "No, it's not a nice review"
    return render_template("home.html", review_check= "{}".format(answer))


if __name__=="__main__":
    app.run(debug=True)




