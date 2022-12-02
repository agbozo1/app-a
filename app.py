import streamlit as st
import joblib
from urllib.request import urlopen


import pickle
pickled = open('model/IrisClassifier.pkl', 'rb')
iris_model = pickle.load(pickled)

#iris_model = joblib.load(urlopen("https://github.com/agbozo1/app-a/blob/main/model/IrisClassifier.pkl"))

#with open('model/IrisClassifier.pkl', 'rb') as handle:
#    iris_model = pickle.loads(handle.read())


layout = 'centered' #wide / centered
page_title = 'Iris App'
page_icon = ':rose:'

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

st.title(page_icon+' '+ page_title)
#html reset
html_style = """ 
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
"""
st.markdown(html_style, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Iris Classification App")


with st.container():
    with st.form(key='my_form'):
        st.text("Enter Details Below")
        sep_len = st.text_input(label='Sepal Length')

        sep_wid = st.text_input(label='Sepal Width')

        pet_len = st.text_input(label='Petal Length')

        pet_wid = st.text_input(label='Petal Width')        

        submitted = st.form_submit_button(label='Predict')
    
    if submitted:
        data = [[int(sep_len), int(sep_wid), int(pet_len), int(pet_wid)]]
        #st.text(sep_len + " " + sep_wid + " " + pet_len + pet_wid)
        #mydata
        st.text(data)

        PredictedFlowers = iris_model.predict(data)
        st.text(PredictedFlowers[0])
        if PredictedFlowers == 'Iris-setosa':           
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/413px-Irissetosa1.jpg")
            
        elif PredictedFlowers == 'Iris-versicolor':
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/800px-Iris_versicolor_3.jpg")

        else:
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/330px-Iris_virginica_2.jpg")
