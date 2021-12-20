
import pandas as pd
import numpy as n
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as tsp
from sklearn import metrics
from flask import Flask, request, jsonify, render_template,url_for,redirect
import time

app = Flask(__name__)

df=pd.read_csv('malariaDataset.csv.txt')

y=df.Label
x=df.drop('Label',axis=1)

x_tr,x_ts,y_tr,y_ts=tsp(x,y,test_size=0.2)


@app.route('/')
def cancerhome():
    return render_template('homepage.html')

@app.route('/predictmalaria',methods=['POST'])
def predictmalaria():
    flacipram=float(request.form['0'])
    vivax=float(request.form['1'])
    oval =float(request.form['2'])
    plasmodium =float(request.form['3'])
    insulin =float(request.form['4'])
    blood_cells_values=[[flacipram,vivax,oval,plasmodium,insulin]]
    
    #starting time
    s_time = time.ctime()
    starting_time=s_time[10:19]
   
   #RFC Algorithm
    clf = rfc()
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_ts)
    
    #convert normal array to numpy array
    final_features = (n.array(blood_cells_values))
    
    #predicted output
    prediction = clf.predict(final_features)[0]
    
    
    #accuracy
    accu = (metrics.accuracy_score(y_ts, y_pred))
    output = round(accu, 4)

    e_time = time.ctime()
    end_time=e_time[10:19]

    if(prediction == 'Parasitized'):
        return render_template('parasitized.html', alg='Random Forest', prediction_text=prediction, acc=output * 100,start_time=starting_time, end_time=end_time)
    else:
        return render_template('uninfected.html', alg='Random Forest', prediction_text=prediction, acc=output * 100, start_time=starting_time, end_time=end_time)
    

if __name__ == "__main__":
    app.run(debug=True)