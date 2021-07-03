#import relevant libraries for flask, html rendering and loading the ML model
from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import *


app = Flask(__name__)

#loading the SVM model and the preprocessor
model = pickle.load(open("svm_model.pkl", "rb"))
#std = pickle.load(open('std.pkl','rb'))
#model = load_model('lstm_model.h5')

#Index.html will be returned for the input
@app.route('/')
def hello_world():
    return render_template("index.html")


#predict function, POST method to take in inputs
@app.route('/predict',methods=['POST','GET'])
def predict():

    #take inputs for all the attributes through the HTML form
    AvgTemp     = request.form['1']
    WetBulbTemp = request.form['2']
    MaxTemp     = request.form['3']
    AirPres     = request.form['4']
    WinSpeed9pm = request.form['5']
    MinTemp     = request.form['6']
    MaxWinSpeed = request.form['7']
    Temp9pm     = request.form['8']
    Humid3pm    = request.form['9']
    DewPoint    = request.form['10']
    RHumid      = request.form['11']
    WindSpeed3pm = request.form['12']
 
 

    #form a dataframe with the inpus and run the preprocessor as used in the training 
    row_df = pd.DataFrame([pd.Series([AvgTemp, WetBulbTemp, MaxTemp, AirPres, WinSpeed9pm, MinTemp, MaxWinSpeed, Temp9pm,Humid3pm,DewPoint,RHumid,WindSpeed3pm])])
#   row_df =  pd.DataFrame(std.transform(row_df))
    scaler = StandardScaler()
    row_df = scaler.fit_transform(row_df)
    row_df = row_df.reshape((row_df.shape[0],1, row_df.shape[1]))
    print(row_df)

    
    prediction=model.predict_proba(row_df)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)
    output = prediction
    #output_print = str(float(output)*100)+'%'
    #output_print = str(float(output)
#   if float(output)>0.5:
#        return render_template('result.html',pred=f'Tomorrow a chance of having Rainfall.\nProbability of you being a diabetic is {output_print}.\nEat clean and exercise regularly')
#    else:
#       return render_template('result.html',pred=f' No Rain.\n Probability of you being a rainfall is {output_print}')
    if float(output)>0.2:
         return render_template('result.html',pred=f'Tomorrow Is Going to Be Rainy Day')
    else:
        return render_template('result.html',pred=f'Tomorrow Is Going to Be Sunny Day')
    #return render_template("predictor.html")

    
if __name__ == '__main__':
    app.run(debug=True)
