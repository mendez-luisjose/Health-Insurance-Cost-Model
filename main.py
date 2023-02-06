import streamlit as st
import pandas as pd
import pickle 
import numpy as np

@st.cache
def get_data() :
    dataset = pd.read_csv("./insurance.csv")

    return dataset

def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pkl('xgb_model.pkl') 

def region_onehot(region) :
    if region == "Southwest" :
        return [0, 0, 0, 1]
    elif region == "Northeast" :
        return [1, 0, 0, 0]
    elif region == "Northwest" :
        return [0, 1, 0, 0]
    elif region == "Southeast" :
        return [0, 0, 1, 0]

def prediction(row) :
   X = pd.DataFrame([row], columns=["age", "bmi", "children", "smoker", "region_northeast", "region_northwest", "region_southeast", "region_southwest"])
   predicted = model.predict(X)
   st.success("Health Insurance Cost: $ " + str(predicted[0]))


header = st.container()
body = st.container()

with header :
    st.title("Health Insurance Model 🏥")
    st.image("./img.jpg")
    st.header("Linear Regression Health Insurance Model with XGB")

with body :
    dataset = get_data()

    st.write("The Model predicts the Health Insurance Cost of a Person in $ Dollars, using Features like Age, BMI, Number of Children, Region and if the Person is Smoker.")
    st.subheader("Data that the Model was Trained: ")
    st.write(dataset.head(15))

    st.subheader("Check It-out!")
    st.write("Please Fill-In all the options for predict the Health Insurance Cost: ")

    age = st.slider("Age: ", min_value=1, max_value=100, value=50, step=1)
    bmi = st.number_input("BMI: ")
    children = st.slider("Number of Children: ", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Are you Smoker? ", options=["Yes", "No"], index=0)
    region = st.selectbox("Which Region do you live? ", options=["Southwest", "Southeast", "Northwest", "Northeast"], index=0)

    if (smoker == "Yes") :
        smoker = 1
    elif (smoker == "No") :
        smoker = 0

    row = np.array([age, bmi, children, smoker])

    row = np.append(row, region_onehot(region))

    st.button("Predict Cost", on_click=prediction, args=(row,))

    st.write("Be aware that the Health Insurance Cost is going to appear at top of the page!")

    


