import streamlit as st
import warnings
warnings.filterwarnings("ignore")

import os

dirname = os.path.dirname(__file__)
from PIL import Image



# page config
st.set_page_config(page_title="Chamberlain Call Center Calculators", page_icon="https://chamberlaingroup.com/wp-content/uploads/2020/01/cgi-favicon.png")
st.sidebar.success('Select a Calculator above')
st.sidebar.header('Instruction')

st.markdown(
    '''
    <style>
    .main {
        background-color: #fbf8f1;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

header = st.container()
lob_and_freq = st.container()
parameter = st.container()
model_prediction = st.container()
erlangC_prediction = st.container()

with header:
    
    image1 = Image.open('chamberlain-logo.png')
    image2 = Image.open('University_of_Chicago.png')
    left, right = st.columns(2)
    left.image(image1)
    right.image(image2)
    st.header("Call Center Calculator")
    st.markdown("@author: University of Chicago MScA 2022 Capstone Team -- Kaicheng Zhang, Alina (Fanting) Zhou, Sherry (Ruiting) Zha")
    st.subheader('Instruction')
    st.markdown('''This calculator, designed for Chamberlain Group and empowered by machine learning, is capable to manage interchangeable variables and output:  
          * **Headcount Calculator** is able to provide headcount requirements based on an adjusted Service Level target.  
          * **Service Level Calculator** is able to provide a projected Service Level based on headcount restraints. ''')
    st.success("👈🏻Select a calculator from the sidebar to get started!")

    



