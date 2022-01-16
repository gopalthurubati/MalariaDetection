import os
import pandas as pd
import numpy as n
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as tsp
from flask import Flask, request, jsonify, render_template,url_for,redirect
import pickle
from sklearn.tree import DecisionTreeClassifier as dtf
from sklearn.svm import SVC
import time
from sklearn import metrics


df=pd.read_csv('malariaDataset.csv.txt')

y=df.Label
x=df.drop('Label',axis=1)
x_tr,x_ts,y_tr,y_ts=tsp(x,y,test_size=0.3)

app = Flask(__name__)
@app.route('/')
def cancerhome():
    return render_template('homepage.html')

def decesiontree():
    clf = dtf()
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_ts)
    
@app.route('/predictmalaria',methods=['POST'])
def predictmalaria():
    flacipram=float(request.form['0'])
    vivax=float(request.form['1'])
    oval=float(request.form['2'])
    plasmodium=float(request.form['3'])
    insulin=float(request.form['4'])
    p=request.form['selection']
    blood_cells_values=[[flacipram,vivax,oval,plasmodium,insulin]]
    
#decision tree
    if (p =='dt'):
        lt1 = time.ctime()
        xx = ''
        s = ''
        for i in range(10, 19):
            s = s + lt1[i]
            
        clf = dtf()
        clf.fit(x_tr, y_tr)
        y_pred = clf.predict(x_ts)
        final_features = (n.array(blood_cells_values))
        prediction = clf.predict(blood_cells_values)
        accu = (metrics.accuracy_score(y_ts, y_pred))
        output = round(accu,3)
        
        lt2 = time.ctime()
        for i in range(10, 19):
            xx = xx + lt2[i]
            
        prediction = (",".join(prediction))
        if (prediction == 'Parasitized'):
            return render_template('parasitized.html', alg='Decesion Tree', prediction_text=prediction,acc=output * 100,s=s, xx=xx)
        else:
            return render_template('uninfected.html', alg='Decision Tree', prediction_text=prediction,acc=output * 100,s=s, xx=xx)


# random forest tree

    elif(p == 'rf'):
            lt1 = time.ctime()
            xx = ''
            s = ''
            for i in range(10, 19):
                s = s + lt1[i]
                
            clf = rfc()
            clf.fit(x_tr, y_tr)
            y_pred = clf.predict(x_ts)
            final_features = (n.array(blood_cells_values))
            prediction = clf.predict(final_features)
            accu = (metrics.accuracy_score(y_ts, y_pred))
            # int_features = [[float(x) for x in request.form.values()]]
            output = round(accu, 3)
            
            #return (str(output))
            lt2 = time.ctime()
            for i in range(10, 19):
                xx = xx + lt2[i]
            prediction = (",".join(prediction))
            if(prediction == 'Parasitized'):
                return render_template('parasitized.html', alg='Random Forest', prediction_text=prediction, acc=output * 100,s=s, xx=xx)
            else:
                return render_template('uninfected.html', alg='Random Forest', prediction_text=prediction, acc=output * 100, s=s,xx=xx)

# svm

    elif (p == 'svm'):
        lt1 = time.ctime()
        xx = ''
        s = ''
        for i in range(10, 19):
            s = s + lt1[i]
            
        clf = SVC()
        clf.fit(x_tr, y_tr)
        y_pred = clf.predict(x_ts)
        final_features = (n.array(blood_cells_values))
        prediction = clf.predict(final_features)
        accu = (metrics.accuracy_score(y_ts, y_pred))
        output = round(accu, 3)
        
        lt2 = time.ctime()
        for i in range(10, 19):
            xx = xx + lt2[i]
        prediction = (",".join(prediction))
        #return prediction
        if(prediction == 'Parasitized'):
            return render_template('parasitized.html', alg='Support Vector Machine', prediction_text=prediction, acc=output * 100, s=s,xx=xx)
        else:
            return render_template('uninfected.html', alg='Support Vector Machine', prediction_text=prediction, acc=output * 100, s=s,xx=xx)





if __name__ == "__main__":
    app.run(debug=True)
