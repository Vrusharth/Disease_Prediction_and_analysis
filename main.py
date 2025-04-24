import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import anthropic
import base64

import PyPDF2

import os
import tempfile

def diabetes(file):

    import os
    # Function to extract text from PDF
    def pdfextract(pdf_path):
        with open(pdf_path, "rb") as pdf:
            reader = PyPDF2.PdfReader(pdf, strict=False)
            text = []
            content = reader.pages[0].extract_text()
            text.append(content)
            return text

    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        pdf_path = tmp_file.name

    pdftext = pdfextract(pdf_path)
    os.unlink(pdf_path)  # Delete the temporary file after extraction

    
    
    prompt = " read the text provided and then provided output as {'age': as per data provided, 'hypertension': as per data provided, 'heart_disease': as per data provided, 'smoking_history': as per data provided, 'bmi': as per data provided, 'blood_glucose_level': as per data provided} where age as Age , hypertension is hypertension, heart_disease as Heart Disease, smoking_history as Smoking history, bmi as bmi and blood_glucose_level as Blood glucose level"

    from fastapi import FastAPI
    from pydantic import BaseModel
    import os
    import google.generativeai as genai

    from dotenv import load_dotenv




  

    
    
    app= FastAPI()

    API_KEY = 'AIzaSyBrX0GtL5TqwCaVKQEldpQnlH2DVdFVX4I'

    genai.configure(
    api_key = API_KEY
    )

    model = genai.GenerativeModel(model_name="models/gemini-pro")
    chat=model.start_chat(history=[])


    response = model.generate_content(prompt + pdftext[0])  # Correct method for single response
    # st.write(response.text)
    
    p = response.text
    

    input_tuple = eval(p)

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    # Load the selected dataset
    data = pd.read_csv("diabetes.csv")
    # Separate features and target variable
    X = data.drop(columns=['diabetes'])
    y = data['diabetes']
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Preprocessing pipeline
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    # Define the logistic regression model
    logistic_regression = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    # Fit the model on training data
    logistic_regression.fit(X_train, y_train)

    input_df = pd.DataFrame([input_tuple])

    # Preprocess the input data
    input_preprocessed = logistic_regression.named_steps['preprocessor'].transform(input_df)

    # Make predictions
    probabilities = logistic_regression.named_steps['classifier'].predict_proba(input_preprocessed)

    # # Print the probability of the first class
    # print("Probability of the first class:", probabilities[0][0])


    if probabilities[0][0] > 0.7:
        st.header('High Probablity of having Diabetes')
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        st.write("( bmi = body mass index)")

    elif probabilities[0][0] >0.3:
        st.header('Moderate Probablity of having Diabetes')
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        st.write("( bmi = body mass index)")

    else:
        st.header("Low Probablity of having Diabetes")
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        st.write("( bmi = body mass index)")

def liver(file):
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    # Load the selected dataset
    data = pd.read_csv("liver.csv")
    # Separate features and target variable
    X = data.drop(columns=['Dataset'])
    y = data['Dataset']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Define the logistic regression model
    logistic_regression = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # Fit the model on training data
    logistic_regression.fit(X_train, y_train)

    # Function to extract text from PDF
    def pdfextract(pdf_path):
        with open(pdf_path, "rb") as pdf:
            reader = PyPDF2.PdfReader(pdf, strict=False)
            text = []
            content = reader.pages[0].extract_text()
            text.append(content)
            return text
    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        pdf_path = tmp_file.name
    pdftext = pdfextract(pdf_path)
    os.unlink(pdf_path)  # Delete the temporary file after extraction

    

    prompt = " read the text provided and then provided output as {'Total_Bilirubin': as per data provided, 'Direct_Bilirubin': as per data provided, 'Alkaline_Phosphotase': as per data provided, 'Alamine_Aminotransferase': as per data provided, 'Aspartate_Aminotransferase': as per data provided, 'Total_Protiens': as per data provided, 'Albumin': as per data provided} where Total_Bilirubin as Total Bilirubin , Direct_Bilirubin is Direct Bilirubin, Alkaline_Phosphotase as Alkaline Phosphotase, Alamine_Aminotransferase as Alamine Aminotransferase, Aspartate_Aminotransferase as Aspartate Aminotransferase, Total_Protiens as Total Protiens and Albumin as Albumin  "

    from fastapi import FastAPI
    from pydantic import BaseModel
    import os
    import google.generativeai as genai

    from dotenv import load_dotenv

    app= FastAPI()

    API_KEY = 'AIzaSyBU1BhybBEaYGRKM45KWvomihSXgYvV22U'

    genai.configure(
    api_key = API_KEY
    )

    model= genai.GenerativeModel('gemini-pro')
    chat=model.start_chat(history=[])


    response=chat.send_message(prompt + pdftext[0])
    # st.write(response.text)
    p = response.text

    input_tuple = eval(p)

    input_df = pd.DataFrame([input_tuple])
    # Preprocess the input data
    input_preprocessed = logistic_regression.named_steps['preprocessor'].transform(input_df)
    # Make predictions
    probabilities = logistic_regression.named_steps['classifier'].predict_proba(input_preprocessed)
    # # Print the probability of the first class
    # print("Probability of the first class:", probabilities[0][0])
    if probabilities[0][0] > 0.7:
        st.header('High Probablity of having Liver Disease')
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        
    elif probabilities[0][0] >0.3:
        st.header('Moderate Probablity of having Liver Disease')
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        
    else:
        st.header("Low Probablity of having Liver Disease")
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        

def heart(file):
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    # Load the selected dataset
    data = pd.read_csv("heart_selected.csv")
    # Separate features and target variable
    X = data.drop(columns=['output'])
    y = data['output']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Define the logistic regression model
    logistic_regression = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # Fit the model on training data
    logistic_regression.fit(X_train, y_train)

    # Function to extract text from PDF
    def pdfextract(pdf_path):
        with open(pdf_path, "rb") as pdf:
            reader = PyPDF2.PdfReader(pdf, strict=False)
            text = []
            content = reader.pages[0].extract_text()
            text.append(content)
            return text
    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        pdf_path = tmp_file.name
    pdftext = pdfextract(pdf_path)
    os.unlink(pdf_path)  # Delete the temporary file after extraction
    prompt = " read the text provided and then provided output as {'age': as per data provided, 'sex': as per data provided, 'cp': as per data provided, 'trtbps': as per data provided, 'chol': as per data provided, 'thalachh': as per data provided, 'caa': as per data provided} where age as Age , sex as Sex, cp as cp, trtbps as trtbps, chol as chol, thalachh as thalachh and caa as caa and only consider numeric values"

    from fastapi import FastAPI
    from pydantic import BaseModel
    import os
    import google.generativeai as genai

    from dotenv import load_dotenv

    app= FastAPI()

    API_KEY = 'AIzaSyBU1BhybBEaYGRKM45KWvomihSXgYvV22U'

    genai.configure(
    api_key = API_KEY
    )

    model= genai.GenerativeModel('gemini-pro')
    chat=model.start_chat(history=[])


    response=chat.send_message(prompt + pdftext[0])
    # st.write(response.text)
    p = response.text

    input_tuple = eval(p)

    

    input_df = pd.DataFrame([input_tuple])
    # Preprocess the input data
    input_preprocessed = logistic_regression.named_steps['preprocessor'].transform(input_df)
    # Make predictions
    probabilities = logistic_regression.named_steps['classifier'].predict_proba(input_preprocessed)
    # # Print the probability of the first class
    # print("Probability of the first class:", probabilities[0][0])
    if probabilities[0][0] > 0.7:
        st.header('High Probablity of having Heart Disease')
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        st.write('(cp: Chest Pain Type , trtbps: Resting Blood Pressure, chol: Serum Cholesterol ,thalachh: Maximum Heart Rate Achieved, caa: Number of Major Vessels (0-3) Colored by Flourosopy)')
    elif probabilities[0][0] >0.3:
        st.header('Moderate Probablity of having Heart Disease')
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        st.write('(cp: Chest Pain Type , trtbps: Resting Blood Pressure, chol: Serum Cholesterol ,thalachh: Maximum Heart Rate Achieved, caa: Number of Major Vessels (0-3) Colored by Flourosopy)')
    else:
        st.header("Low Probablity of having Heart Disease")
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        st.write('(cp: Chest Pain Type , trtbps: Resting Blood Pressure, chol: Serum Cholesterol ,thalachh: Maximum Heart Rate Achieved, caa: Number of Major Vessels (0-3) Colored by Flourosopy)')

def lung(file):
    import os
    # Function to extract text from PDF
    def pdfextract(pdf_path):
        with open(pdf_path, "rb") as pdf:
            reader = PyPDF2.PdfReader(pdf, strict=False)
            text = []
            content = reader.pages[0].extract_text()
            text.append(content)
            return text
        
    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        pdf_path = tmp_file.name

    pdftext = pdfextract(pdf_path)
    os.unlink(pdf_path)  # Delete the temporary file after extraction

    

    prompt = " read the text provided and then provided output as {'SMOKING': as per data provided, 'AGE': as per data provided, 'CHRONIC DISEASE': as per data provided, 'WHEEZING': as per data provided, 'CHEST PAIN': as per data provided, 'SHORTNESS OF BREATH': as per data provided } where SMOKING as SMOKING , AGE is Age, CHRONIC DISEASE as CHRONIC DISEASE, WHEEZING as WHEEZING, CHEST PAIN as CHEST PAIN and SHORTNESS OF BREATH as SHORTNESS OF BREATH also take only numeric values"
    from fastapi import FastAPI
    from pydantic import BaseModel
    import os
    import google.generativeai as genai
    from dotenv import load_dotenv
    app= FastAPI()
    API_KEY = 'AIzaSyBU1BhybBEaYGRKM45KWvomihSXgYvV22U'
    genai.configure(
    api_key = API_KEY
    )
    model= genai.GenerativeModel('gemini-pro')
    chat=model.start_chat(history=[])
    response=chat.send_message(prompt + pdftext[0])
    # st.write(response.text)
    p = response.text
    input_tuple = eval(p)

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    # Load the selected dataset
    data = pd.read_csv("lung.csv")
    columns=['LUNG_CANCER']
    # Separate features and target variable
    X = data.drop(columns=['LUNG_CANCER'])
    y = data['LUNG_CANCER']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Define the logistic regression model
    logistic_regression = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # Fit the model on training data
    logistic_regression.fit(X_train, y_train)


    input_df = pd.DataFrame([input_tuple])
    # Preprocess the input data
    # Convert dictionary to DataFrame
    

    # Predict the probability of lung cancer
    probabilities = logistic_regression.predict_proba(input_df)[:, 1]
    # # Print the probability of the first class
    # print("Probability of the first class:", probabilities[0][0])
    if probabilities[0] > 0.7:
        st.header('High Probablity of having Lung Disease')
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
    elif probabilities[0] >0.3:
        st.header('Moderate Probablity of having Lung Disease')
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
    else:
        st.header("Low Probablity of having Lung Disease")
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")

def kidney(file):
    import os
    # Function to extract text from PDF
    def pdfextract(pdf_path):
        with open(pdf_path, "rb") as pdf:
            reader = PyPDF2.PdfReader(pdf, strict=False)
            text = []
            content = reader.pages[0].extract_text()
            text.append(content)
            return text

    # Save the uploaded PDF file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        pdf_path = tmp_file.name

    pdftext = pdfextract(pdf_path)
    os.unlink(pdf_path)  # Delete the temporary file after extraction


    prompt = " read the text provided and then provided output as {'sc': as per data provided, 'bu': as per data provided, 'bgr': as per data provided, 'hemo': as per data provided, 'htn': as per data provided, 'dm': as per data provided, 'rc': as per data provided} where sc as serum creatinine , bu is blood urea, bgr as glucose, hemo as hemoglobin, htn as hypertension, dm as diabetes multiuse and rc as red blood cells  "

    from fastapi import FastAPI
    from pydantic import BaseModel
    import os
    import google.generativeai as genai

    from dotenv import load_dotenv

    app= FastAPI()

    API_KEY = 'AIzaSyBU1BhybBEaYGRKM45KWvomihSXgYvV22U'

    genai.configure(
    api_key = API_KEY
    )

    model= genai.GenerativeModel('gemini-pro')
    chat=model.start_chat(history=[])


    response=chat.send_message(prompt + pdftext[0])
    # st.write(response.text)
    
    p = response.text
    

    input_tuple = eval(p)



    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    # Load the selected dataset
    data = pd.read_csv("kidney_disease_selected.csv")

    # Separate features and target variable
    X = data.drop(columns=['classification'])
    y = data['classification']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Define the logistic regression model
    logistic_regression = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # Fit the model on training data
    logistic_regression.fit(X_train, y_train)


    input_df = pd.DataFrame([input_tuple])

    # Preprocess the input data
    input_preprocessed = logistic_regression.named_steps['preprocessor'].transform(input_df)

    # Make predictions
    probabilities = logistic_regression.named_steps['classifier'].predict_proba(input_preprocessed)

    # # Print the probability of the first class
    # print("Probability of the first class:", probabilities[0][0])


    if probabilities[0][0] > 0.7:
        st.header('High Probablity of having Kidney Disease')
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        st.write(' sc : Serum Creatinine, bu :  Blood Urea , bgr :  Blood Glucose Random , hemo :  Hemoglobin , htn :  Hypertension , dm :  Diabetes Mellitus , rc :  Red Blood Cell Count')

    elif probabilities[0][0] >0.3:
        st.header('Moderate Probablity of having Kidney Disease')
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        st.write(' sc : Serum Creatinine, bu :  Blood Urea , bgr :  Blood Glucose Random , hemo :  Hemoglobin , htn :  Hypertension , dm :  Diabetes Mellitus , rc :  Red Blood Cell Count')

    else:
        st.header("Low Probablity of having Kidney Disease")
        st.write("These factors can be responsible according to report :")
        for key, value in input_tuple.items():
            st.write(f"{key}: {value}")
        st.write(' sc : Serum Creatinine, bu :  Blood Urea , bgr :  Blood Glucose Random , hemo :  Hemoglobin , htn :  Hypertension , dm :  Diabetes Mellitus , rc :  Red Blood Cell Count')





# Streamlit UI
st.title('Disease Risk Prediction')

# Input for patient name


# File uploader for PDF selection
uploaded_file = st.file_uploader("Choose a PDF file...", type=["pdf"])


# Define the options
options = ['Kidney', 'Heart', 'Liver', 'Lung', 'Diabetes']

# Create the dropdown
selected_option = st.selectbox('Select an option:', options)

# Display different buttons based on the selected option
if selected_option == 'Kidney':
    if st.button('Predict Kidney Disease'):
        kidney(uploaded_file)
elif selected_option == 'Heart':
    if st.button('Predict Heart Disease'):
        heart(uploaded_file)
elif selected_option == 'Liver':
    if st.button('Predict Liver Disease'):
        liver(uploaded_file)
elif selected_option == 'Lung':
    if st.button('Predict Lung Disease'):
        lung(uploaded_file)
elif selected_option == 'Diabetes':
    if st.button('Predict Diabetes '):
        diabetes(uploaded_file)

