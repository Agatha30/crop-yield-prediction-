import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from PIL import Image as img
from tkinter import Image

#img=('MILLET.jpg')  # type: ignores

#st.image(img, width=350)

train_df = pd.read_csv(r'C:\Users\Ropa\Desktop\project\yield3.csv')
train_df.dropna(inplace=True)
print(train_df.head())

crop_data = pd.get_dummies(train_df['Crop_Name'], drop_first = True)
print(crop_data)
train_df = pd.concat([train_df, crop_data], axis = 1)
print(train_df)

#train_df.drop(['Crop_Name'], axis = 1, inplace = True)
#print(train_df)

features= train_df[['Rainfall', 'Humidity', 'Temperature', 'Pesticides', 'Soil_ph', 
'N','P','K','Area_Planted']]
tests = train_df['Maize']
X_train , X_test , y_train , y_test = train_test_split(features , tests ,test_size = 0.3)

scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(train_features , y_train)
train_score = model.score(train_features,y_train)
test_score = model.score(test_features,y_test)
y_predict = model.predict(test_features)
print(train_score)
print(test_score)
print(y_predict)
import pickle

# save the model to disk
filename = 'logistic_reg.pkl'
pickle.dump(filename, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))




st.markdown("")

html_temp = """
<div style = "background-color:green; padding: 14px">
<h1 style = "color: black; text-align: centre; "> CROP YIELD PREDICTION APP--ML
</h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

Place = st.selectbox('DISTRICT',('Gwanda','Umzingwane','Insiza','Matobo','Beitbridge'))

Crop_Name = st.selectbox('CROP NAME',('Maize','Sorghum','Cowpeas','FingerMillet','Soyabeans','Wheat',
'Groundnut','Sweet Potato','Sugarbean','Tomatoes','Cabbage','Black pepper','Irish Potato','Onion',
'Butternut','Sunflower','Carrot','Pumpkin'))

Rainfall = st.slider("RAINFALL (MM)", 1, 5000,1)
Humidity = st.slider("HUMIDITY (%)", 1, 100,1)
Temperature = st.slider("TEMPERATURE (Degrees Celcius)", 1, 100,1)
Pesticides = st.slider("PESTICIDES (Tonnes)", 1, 5000,1)
Soil_ph = st.slider("SOIL PH", 1, 14,1)
N = st.slider("Nitrogen", 1, 500,1)
P = st.slider("Phosphorous", 1, 500,1)
K = st.slider("Potassium", 1, 500,1)
Area_Planted = st.slider("Area Planted (Hectares)", 1, 50000,1)

safe_html="""
<div style="background-color:green;padding:10px">
<h2 style="color:white:text-align-centre:">Crop Yield is High</h2>
</div>
"""

danger_html="""
<div style="background-color:#880808;padding:10px">
<h2 style="color:black:text-align-centre:">Crop Yield is LOW</h2>
</div>
"""


if st.button('Predict '):
   input_data = scaler.transform([[Rainfall,Humidity,Temperature,Pesticides,Soil_ph, N,P,K,Area_Planted]])
   prediction = model.predict(input_data)
   predict_yieldProduction = prediction 
   predict_yieldProduction = model.predict_proba(input_data)*4.5
   
   yield_per_hectare = predict_yieldProduction/Area_Planted

   st.subheader('{} Crop in {} District has Total Production Yield: {} tonnes'.format(Crop_Name, Place , round(predict_yieldProduction[0][1]*55000, 3)))
   st.subheader('{} Yield Per Hectare: {} tonnes'.format(Crop_Name, round(predict_yieldProduction[0][1]/(Area_Planted)*55000, 3)))
   #st.subheader('Yield per hectare is {} tonnes'.format( round(yield_per_hectare, 3)))
   
   #if predict_yieldProduction[0][1]*100 < 20:
   if yield_per_hectare [0][1]*55000 < 10.0:
       #st.subheader('Predicted yield is low')
       st.markdown(danger_html,unsafe_allow_html=True)
	  
   else:
        #st.subheader('Predicted yield is high')
        st.markdown(safe_html,unsafe_allow_html=True)
