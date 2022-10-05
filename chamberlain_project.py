import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stat
# Model
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

# Evaulation Metrics
from sklearn.metrics import mean_squared_error
import joblib

import os

dirname = os.path.dirname(__file__)


#load models
#parameters  ['Headcount', 'Line_of_Business', 'Call Volume', 'Average Handle Time',
# 'Average Speed of Answer', 'Occupany %', 'Shrinkage %']
#'Line_of_Business': 0 for COMM, 1 for RESI
@st.cache
def load_sl():
    filename1 = os.path.join(dirname, 'sl.joblib')
    model_sl = joblib.load(filename1)
    return model_sl
#parameters  ['Service Level', 'Line_of_Business', 'Call Volume', 'Average Handle Time',
# 'Average Speed of Answer', 'Occupany %', 'Shrinkage %']
#'Line_of_Business': 0 for COMM, 1 for RESI
@st.cache
def load_hc():
    filename2 = os.path.join(dirname, 'hc.joblib')
    model_hc = joblib.load(filename2)
    return model_hc

#load RMSE
RMSE_sl = 0.055390367887275326
RMSE_hc = 7.99815336781414

def hc_prediction(param,CI):
    alpha = 1 - CI
    upper = alpha/2
    Z = stat.norm.ppf(1-upper)          #Above three line for Calculating Z-score
    param = param.reshape(1,-1)
    #print(param)
    model_hc = load_hc()
    result = model_hc.predict(param)[0]    #Prediction of HC
    #print(result)
    #result = float(str(result)[1:-1])      #Removing bracket from array results
    low = result - Z* RMSE_hc                #Upper and Lower bound
    high = result + Z * RMSE_hc
    return result, low, high

def get_HC(param, CI):
    result, low, high = hc_prediction(param, CI)
    return result


def get_HC_low(param, CI):
    result, low, high = hc_prediction(param, CI)
    return low

def get_HC_high(param, CI):
    result, low, high = hc_prediction(param, CI)
    return high

def adjust_percentage(percentage):
    if percentage > 1:
        percentage = 1
    if percentage < 0:
        percentage = 0
    return percentage



def sl_prediction(param,CI):
    alpha = 1 - CI
    upper = alpha/2
    Z = stat.norm.ppf(1-upper)          #Above three line for Calculating Z-score
    model_sl = load_sl()
    result = model_sl.predict(param.reshape(1, -1))[0]    #Prediction of SL
    #result = float(str(result)[1:-1])      #Removing bracket from array results
    low = result - Z* RMSE_sl                #Upper and Lower bound
    high = result + Z * RMSE_sl
    return result, low, high


def get_SL(param, CI):
    result, low, high = sl_prediction(param, CI)
    if result > 1:
        result = 1
    if result < 0:
        result = 0
    return result


def get_SL_low(param, CI):
    result, low, high = sl_prediction(param, CI)
    if low > 1:
        low = 1
    if low < 0:
        low = 0
    return low


def get_SL_high(param, CI):
    result, low, high = sl_prediction(param, CI)
    if high > 1:
        high = 1
    if high < 0:
        high = 0
    return high

header = st.container()
model_display = st.container()

with header:
    st.title("Chamberlain Group Call Center Calculator")
    st.text("@author: UChicago MScA Capstone Team - Kaicheng Zhang, Alina Zhou, Sherry Zha")
    st.markdown("Use this calculator to determine your expected headcount (the number of agents) depending on service level and others.")

with model_display:
    st.header("Headcount Calculator")

    select_col, display_col = st.columns(2)
    select_col.subheader('Enter Parameters')
    #catch necesary parameters
    service_level_text = select_col.text_input('Service Level Target % (e.g. enter 78.9 for 78.9%)', '80.0')
    try:
        service_level = float(service_level_text)/100
    except:
        st.write('Unexpected entry. Try again.')
        service_level = 0.8
    
    line_of_business_text = select_col.selectbox('Line of Business', options = ['COMM', 'RESI'], index = 0)
    if line_of_business_text == 'COMM':
        lob = 0
    elif line_of_business_text == 'RESI':
        lob = 1
    else:
        lob = 0
    
    call_volume_text = select_col.text_input('Day Calls Volumes', '2000')
    try:
        call_volume = float(call_volume_text)
    except:
        st.write('Unexpected entry. Try again.')
        call_volume = 2000
    
    average_handle_text = select_col.text_input('Average Handling Time for Interactions in Seconds', '700')
    try:
        aht = float(average_handle_text)
    except:
        st.write('Unexpected entry. Try again.')
        aht = 700
    
    average_speed_text = select_col.text_input('Average Speed of Answer', '100')
    try:
        apa = float(average_speed_text)
    except:
        st.write('Unexpected entry. Try again.')
        apa = 100

    occupancy_text = select_col.text_input('Maximum Occupany % (e.g. enter 78.9 for 78.9%)', '80.0')
    try:
        occupancy = float(occupancy_text)/100
    except:
        st.write('Unexpected entry. Try again.')
        occupancy = 0.8
    
    shrinkage_text = select_col.text_input('Shrinkage % (e.g. enter 78.9 for 78.9%)', '30.0')
    try:
        shrinkage = float(shrinkage_text)/100
    except:
        st.write('Unexpected entry. Try again.')
        shrinkage = 0.8
    
    confidence_level = select_col.slider('Confidence Level %', min_value = 80, max_value = 99, step = 1)/100

    param = np.array([service_level, lob, call_volume, aht, apa, occupancy, shrinkage])

    # display the results
    display_col.subheader('Results')
    display_col.markdown('**HeadCount**')
    display_col.write(round(get_HC(param, confidence_level), 1))

    display_col.markdown('**Lower Bound of Confidence Interval**')
    display_col.write(round(get_HC_low(param, confidence_level), 1))

    display_col.markdown('**Upper Bound of Confidence Interval**')
    display_col.write(round(get_HC_high(param, confidence_level), 1))
    


    



