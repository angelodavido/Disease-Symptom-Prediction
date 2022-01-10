from flask import Flask, render_template, request
import pickle
import numpy as np 
from tensorflow import keras
import pandas as pd
import os
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


# label_encoder = preprocessing.LabelEncoder()
#Create a scaler model that is fit on the input data.
scaler = StandardScaler()
#Loading a Model 
loaded_model = keras.models.load_model("C:/Users/mwamb/Desktop/Desktop/disease prediction/disease_save")
# loaded_model =  tf.keras.models.load_model("C:/Users/mwamb/Desktop/Desktop/disease prediction/disease_save/saved_model.pb")

disease_dict = { 'Fungal infection':15, 'Allergy':4, 'GERD':16, 'Chronic cholestasis':9,
       'Drug Reaction':14, 'Peptic ulcer diseae':33, 'AIDS':1, 'Diabetes ':12,
       'Gastroenteritis':17, 'Bronchial Asthma':6, 'Hypertension ':23, 'Migraine':30,
       'Cervical spondylosis':7, 'Paralysis (brain hemorrhage)':32, 'Jaundice':28,
       'Malaria':29, 'Chicken pox':8, 'Dengue':11, 'Typhoid':37, 'hepatitis A':40,
       'Hepatitis B':19, 'Hepatitis C':20, 'Hepatitis D':21, 'Hepatitis E':22,
       'Alcoholic hepatitis':3, 'Tuberculosis':36, 'Common Cold':10, 'Pneumonia':34,
       'Dimorphic hemmorhoids(piles)':13, 'Heart attack':18, 'Varicose veins':39,
       'Hypothyroidism':26, 'Hyperthyroidism':24, 'Hypoglycemia':25,
       'Osteoarthristis':31, 'Arthritis':5,
       '(vertigo) Paroymsal  Positional Vertigo':0, 'Acne':2,
       'Urinary tract infection':38, 'Psoriasis':35, 'Impetigo':27}
# df = pd.DataFrame(columns=["symptom1", "symptom2", "symptom3", "symptom4", "symptom5", "symptom6", "symptom7", "symptom8", "symptom9", "symptom10", "symptom11", "symptom12", "symptom13", "symptom14", "symptom15", "symptom16" "symptom17"])
# df

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def home():
    df = pd.DataFrame(columns=["symptom1", "symptom2", "symptom3", "symptom4", "symptom5", "symptom6", "symptom7", "symptom8", "symptom9", "symptom10", "symptom11", "symptom12", "symptom13", "symptom14", "symptom15", "symptom16", "symptom17"])

    data1 = request.form['symptom1']
    data2 = request.form['symptom2']
    data3 = request.form['symptom3']
    data4 = request.form['symptom4']
    data5 = request.form['symptom5']
    data6 = request.form['symptom6']
    data7 = request.form['symptom7']
    data8 = request.form['symptom8']
    data9 = request.form['symptom9']
    data10 = request.form['symptom10']
    data11 = request.form['symptom11']
    data12 = request.form['symptom12']
    data13 = request.form['symptom13']
    data14 = request.form['symptom14']
    data15 = request.form['symptom15']
    data16 = request.form['symptom16']
    data17 = request.form['symptom17']

    df1 = pd.DataFrame(data=[[data1,data2,data3,data4,data5,data6, data7, data8,data9,data10,data11,data12, data13,data14,data15,data16,data17]],columns=["symptom1", "symptom2", "symptom3", "symptom4", "symptom5", "symptom6", "symptom7", "symptom8","symptom9", "symptom10", "symptom11", "symptom12", "symptom13", "symptom14", "symptom15", "symptom16", "symptom17"])
    df = pd.concat([df,df1], axis=0)
    print(df)

    df["symptom1"] = df["symptom1"].astype('category')
    df["symptom2"] = df["symptom2"].astype('category')
    df["symptom3"] = df["symptom3"].astype('category')
    df["symptom4"] = df["symptom4"].astype('category')
    df["symptom5"] = df["symptom5"].astype('category')
    df["symptom6"] = df["symptom6"].astype('category')
    df["symptom7"] = df["symptom7"].astype('category')
    df["symptom8"] = df["symptom8"].astype('category')
    df["symptom9"] = df["symptom9"].astype('category')
    df["symptom10"] = df["symptom10"].astype('category')
    df["symptom11"] = df["symptom11"].astype('category')
    df["symptom12"] = df["symptom12"].astype('category')
    df["symptom13"] = df["symptom13"].astype('category')
    df["symptom14"] = df["symptom14"].astype('category')
    df["symptom15"] = df["symptom15"].astype('category')
    df["symptom16"] = df["symptom16"].astype('category')
    df["symptom17"] = df["symptom17"].astype('category')
    # print(df.dtypes)


    df["symptom01"] = df["symptom1"].cat.codes
    df["symptom02"] = df["symptom2"].cat.codes
    df["symptom03"] = df["symptom3"].cat.codes
    df["symptom04"] = df["symptom4"].cat.codes
    df["symptom05"] = df["symptom5"].cat.codes
    df["symptom06"] = df["symptom6"].cat.codes
    df["symptom07"] = df["symptom7"].cat.codes
    df["symptom08"] = df["symptom8"].cat.codes
    df["symptom09"] = df["symptom9"].cat.codes
    df["symptom010"] = df["symptom10"].cat.codes
    df["symptom011"] = df["symptom11"].cat.codes
    df["symptom012"] = df["symptom12"].cat.codes
    df["symptom013"] = df["symptom13"].cat.codes
    df["symptom014"] = df["symptom14"].cat.codes
    df["symptom015"] = df["symptom15"].cat.codes
    df["symptom016"] = df["symptom16"].cat.codes
    df["symptom017"] = df["symptom17"].cat.codes  
    # print(df.dtypes) 

    df.drop(['symptom1','symptom2','symptom3','symptom4','symptom5','symptom6','symptom7','symptom8','symptom9','symptom10','symptom11','symptom12','symptom13','symptom14','symptom15','symptom16','symptom17'], axis = 1, inplace = True)
    # print(df.head())

    #Convert input to numpy array
    np_df = df.to_numpy()
    # print(np_df)

    scaler.fit(np_df)

        #Scale prediction data with the same scaling model
    scaled_input = scaler.transform(np_df)

    #Get raw prediction probabilities
    raw_prediction = loaded_model.predict(scaled_input)
    # print("Raw Prediction Output (Probabilities) :" , raw_prediction)

    #Find prediction
    prediction = np.argmax(raw_prediction)

    # pred = label_encoder.inverse_transform([prediction])
    
    for key, val in disease_dict.items():
        if val== prediction:
            print(key)




    return render_template('after.html', key = key)



    
if __name__ == "__main__":
    app.run(debug=True)





