import streamlit as st # type: ignore
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
except Exception as e:
    tf = None
    # If TensorFlow is not available, show an error in the Streamlit app and stop further execution
    import streamlit as _st
    _st.error(f"TensorFlow could not be imported: {e}")
    _st.stop()
from sklearn.preprocessing import StandardScaler , OneHotEncoder , LabelEncoder
import pickle

model = tf.keras.models.load_model("model.h5")



with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gender = pickle.load(file)
with open("onehotencoder_geo.pkl","rb") as file:
    onehotencoder_geo=pickle.load(file)
with open("scaler.pkl","rb")as file:
    scaler=pickle.load(file) 
geography = st.selectbox('Geography', onehotencoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


input_data = pd.DataFrame({
'CreditScore': [credit_score],
'Gender': [label_encoder_gender.transform([gender])[0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_products],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_member],
'EstimatedSalary': [estimated_salary]
})

# One-hot enode 'Geography
geo_encoded = onehotencoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat(
    [input_data.reset_index(drop=True),
     geo_encoded_df.reset_index(drop=True)],
    axis=1
)

input_scaled_data=scaler.transform(input_data)

prediction = model.predict(input_scaled_data)
predictrion_proba = prediction[0][0]


st.write(f"Churn Probablity;{predictrion_proba}")
if predictrion_proba > 0.50:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")