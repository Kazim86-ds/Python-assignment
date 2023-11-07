import streamlit as st
import pandas as pd
import joblib
import time

parameter_list=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean']
parameter_input_values=[]
parameter_description=['Mean of distances from center to points on the perimeter',
'Mean of gray-scale values',
'Mean represents the mean value for the core tumor',
'Mean of Tumor size (mm or cm)',
'Mean of Local variation in radius lengths',
'(perimeter^2 / area - 1.0)',
'Severity of concave portions of the contour',
'Number of concave portions of the contour']
parameter_default_values=['13.72','18.83','89.05','580.09','0.1','0.09','0.06','0.03']

with st.spinner('Fetching Latest ML Model'):
    # Use pickle to load in the pre-trained model
    model = joblib.load("Breast_Cancer.pkl")
    time.sleep(1)
    st.success('Model V1 Loaded!')


st.title('Breast Cancer Prediction App \n\n')

for parameter,parameter_df,parameter_desc in zip(parameter_list,parameter_default_values,parameter_description):
    #print (parameter,parameter_df,parameter_desc)
    st.subheader('Input value for '+parameter)
    parameter_input_values.append(st.number_input(parameter_desc,key=parameter,value=float(parameter_df)))
        
parameter_dict=dict(zip(parameter_list, parameter_input_values)) 

st.write('\n','\n')
st.title('Your Input Summary')

st.write(parameter_dict)

st.write('\n','\n')

def predict(input_predict,feature_names):
    values = input_predict['data'] 

    input_variables = pd.DataFrame([values], 
                                columns=feature_names, 
                                dtype=float,
                                index=['input'])    
    
    # Get the model's prediction
    prediction = model.predict(input_variables)
    print("Prediction: ", prediction)
    prediction_proba = model.predict_proba(input_variables)[0][1]
    print("Probabilities: ", prediction_proba)

    ret = {"prediction":float(prediction),"prediction_proba": float(prediction_proba)}
    
    return ret

if st.button("Click Here to Predict"):

    PARAMS={'data':list(parameter_dict.values())}
    
    r=predict(PARAMS, parameter_list)
    
    st.write('\n','\n')
    
    prediction_proba=r.get('prediction_proba')
    prediction_proba_format = str(round(float(r.get('prediction_proba')),1)*100)+'%'
    
    prediction_value=r.get('prediction')
    
    prediction_bool='Positive' if float(prediction_proba) > 0.4 else 'Negative'
    
    st.write(f'Your Breast Cancer Prediction is: **{prediction_bool}** with **{prediction_proba_format}** confidence')
