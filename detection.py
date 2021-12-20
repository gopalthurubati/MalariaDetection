import os
import pandas as pd
import numpy as n
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as tsp
from sklearn import metrics
from flask import Flask, request, jsonify, render_template,url_for,redirect

app = Flask(__name__)
@app.route('/')
def cancerhome():
    return render_template('homepage.html')

df=pd.read_csv('malariaDataset.csv.txt')


y=df.Label
x=df.drop('Label',axis=1)

x_tr,x_ts,y_tr,y_ts=tsp(x,y,test_size=0.2)
clf=rfc()

clf.fit(x_tr,y_tr)

y_pred=clf.predict(x_ts)


print(metrics.accuracy_score(y_ts,y_pred))

n1=float(input())
n2=float(input())
n3=float(input())
n4=float(input())
n5=float(input())


zz=clf.predict(n.array([[n1,n2,n3,n4,n5]]))
print(*zz[0])


if __name__ == "__main__":
    app.run(debug=True)