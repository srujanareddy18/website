
from ast import FormattedValue
import re
from flask import Flask, render_template,request
import pickle
import numpy as np


app= Flask(__name__ , template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('home.html')


#To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = int(prediction[0])
    return render_template('after.html', prediction_text=' {}'.format(output))






if __name__=="__main__":
    app.run(debug=True)