from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from joblib import load


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview')
def preview():
    df = pd.read_csv('/home/Seiawan/mysite/data/ikan.csv')
    return render_template('preview.html', df_view=df)

@app.route('/predict', methods=["POST"])
def analyze():
    if request.method == 'POST':
        panjang_tubuh = request.form['panjang_tubuh']
        lebar_tubuh = request.form['lebar_tubuh']
        panjang_sirip = request.form['panjang_sirip']
        lebar_sirip = request.form['lebar_sirip']
        model_choice = request.form['model_choice']

        # Menyiapkan data untuk prediksi
        sample_data = [panjang_tubuh, lebar_tubuh, panjang_sirip, lebar_sirip]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1, -1)

        # Memuat model dan membuat prediksi berdasarkan pilihan model
        if model_choice == 'Logistic':
            logit_model = load('/home/Seiawan/mysite/data/logit_ikan.pkl')
            result_prediction = logit_model.predict(ex1)
        elif model_choice == 'KNN':
            knn_model = load('/home/Seiawan/mysite/data/knn_ikan.pkl')  
            result_prediction = knn_model.predict(ex1)
        elif model_choice == 'SVM':
            svm_model = load('/home/Seiawan/mysite/data/svm_ikan.pkl')
            result_prediction = svm_model.predict(ex1)
        else:
            result_prediction = "Model tidak ditemukan"

        return render_template('predict.html',
                               panjang_tubuh=panjang_tubuh,
                               lebar_tubuh=lebar_tubuh,
                               panjang_sirip=panjang_sirip,
                               lebar_sirip=lebar_sirip,
                               clean_data=clean_data,
                               result_prediction=result_prediction,
                               model_selected=model_choice)

if __name__ == '__main__':
    app.run(debug=True)