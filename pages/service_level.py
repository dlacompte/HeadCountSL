import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stat
# Model
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

# Evaulation Metrics
from sklearn.metrics import mean_squared_error
import joblib
from math import exp, ceil, floor


import os

dirname = os.path.dirname(__file__)

class MyProgram:
    def __init__(self, transactions: float, aht: float, asa: float,
                 interval: int, shrinkage=0.0, occupancy = 1,
                 **kwargs):

        if transactions <= 0:
            raise ValueError("transactions can't be smaller or equals than 0")

        if aht <= 0:
            raise ValueError("aht can't be smaller or equals than 0")

        if asa <= 0:
            raise ValueError("asa can't be smaller or equals than 0")

        if interval <= 0:
            raise ValueError("interval can't be smaller or equals than 0")

        if shrinkage < 0 or shrinkage >= 1:
            raise ValueError("shrinkage must be between in the interval [0,1)")
            
        if occupancy < 0 or occupancy > 1:
            raise ValueError("occupancy must be between in the interval [0,1)")

        self.n_transactions = transactions
        self.aht = aht
        self.interval = interval
        self.asa = asa
        self.intensity = (self.n_transactions / self.interval) * self.aht
        self.shrinkage = shrinkage
        self.occupancy = occupancy

    def waiting_probability(self, positions: int, scale_positions: bool = False):

        if scale_positions:
            productive_positions = floor((1 - self.shrinkage) * positions)
        else:
            productive_positions = positions

        erlang_b_inverse = 1
        for position in range(1, productive_positions + 1):
            erlang_b_inverse = 1 + (erlang_b_inverse * position / self.intensity)

        erlang_b = 1 / erlang_b_inverse
        return productive_positions * erlang_b / (productive_positions - self.intensity * (1 - erlang_b))
        
    def service_level(self, positions: int, scale_positions: bool = True):
        if scale_positions:
            productive_positions = floor((1 - self.shrinkage) * positions )
        else:
            productive_positions = positions

        probability_wait = self.waiting_probability(productive_positions, scale_positions=False)
        exponential = exp(-(productive_positions - self.intensity) * (self.asa / self.aht))
        return max(0, 1 - (probability_wait * exponential))
    
 
    def achieved_occupancy(self, positions: int, scale_positions: bool = False):
        if scale_positions:
            productive_positions = floor((1 - self.shrinkage) * positions)
        else:
            productive_positions = positions

        return self.intensity / productive_positions
    
    def required_positions(self, service_level: float):
        if service_level < 0 or service_level > 1:
            raise ValueError("service_level must be between 0 and 1")

        positions = round(self.intensity + 1)
        achieved_service_level = self.service_level(positions, scale_positions=False) 
        while achieved_service_level < service_level:
            positions += 1
            achieved_service_level = self.service_level(positions, scale_positions=False)

        achieved_occupancy = self.achieved_occupancy(positions, scale_positions=False)
        raw_positions = ceil(positions)


        positions = ceil(raw_positions / (1 - self.shrinkage))

        return {"positions": positions,
                "service_level": achieved_service_level}

#load models
#parameters  ['Headcount', 'Line_of_Business', 'Call Volume', 'Average Handle Time',
# 'Average Speed of Answer', 'Occupany %', 'Shrinkage %']
#'Line_of_Business': 0 for COMM, 1 for RESI
@st.cache
def load_res_sl():
    filename1 = os.path.join(dirname, 'res_sl.joblib')
    model_sl = joblib.load(filename1)
    return model_sl
#parameters  ['Service Level', 'Line_of_Business', 'Call Volume', 'Average Handle Time',
# 'Average Speed of Answer', 'Occupany %', 'Shrinkage %']
#'Line_of_Business': 0 for COMM, 1 for RESI
@st.cache
def load_myq_sl():
    filename2 = os.path.join(dirname, 'myq_sl.joblib')
    model_sl = joblib.load(filename2)
    return model_sl


#parameters  ['Headcount', 'Call Volume', 'Average Handle Time',
# 'Average Speed of Answer', 'Occupany %', 'Shrinkage %']
@st.cache
def load_com_sl():
    filename4 = os.path.join(dirname, 'com_sl.joblib')
    model_sl = joblib.load(filename4)
    return model_sl

# page config
st.set_page_config(page_title="Service Level Calculator", page_icon="📈")
st.sidebar.success('Select a Calculator above')
st.sidebar.header('Service Level Calculator')


#load RMSE
RMSE_com_sl = 0.042765867920959245
RMSE_myq_sl = 0.057959133768238046
RMSE_res_sl = 0.06071274838102916
columns = ['HC', 'Call Volume','AHT','ASA','Occupany %','Shrinkage %']

def adjust_percentage(percentage):
    if percentage > 1:
        percentage = 1
    if percentage < 0:
        percentage = 0
    return percentage



def res_sl_prediction(param,CI):
    alpha = 1 - CI
    upper = alpha/2
    Z = stat.norm.ppf(1-upper)          #Above three line for Calculating Z-score
    model_sl = load_res_sl()
    param = pd.DataFrame([param], columns=columns)
    result = model_sl.predict(param)[0]    #Prediction of SL
    low = result - Z* RMSE_res_sl                #Upper and Lower bound
    high = result + Z * RMSE_res_sl
    return result, low, high

def com_sl_prediction(param,CI):
    alpha = 1 - CI
    upper = alpha/2
    Z = stat.norm.ppf(1-upper)          #Above three line for Calculating Z-score
    model_sl = load_com_sl()
    param = pd.DataFrame([param], columns=columns)
    result = model_sl.predict(param)[0]    #Prediction of SL
    low = result - Z* RMSE_com_sl                #Upper and Lower bound
    high = result + Z * RMSE_com_sl
    return result, low, high

def myq_sl_prediction(param,CI):
    alpha = 1 - CI
    upper = alpha/2
    Z = stat.norm.ppf(1-upper)          #Above three line for Calculating Z-score
    model_sl = load_myq_sl()
    param = pd.DataFrame([param], columns=columns)
    result = model_sl.predict(param)[0]    #Prediction of SL
    low = result - Z* RMSE_myq_sl                #Upper and Lower bound
    high = result + Z * RMSE_myq_sl
    return result, low, high

def get_SL(param, CI, predictor):
    result, low, high = predictor(param, CI)
    if result > 1:
        result = 1
    if result < 0:
        result = 0
    return result


def get_SL_low(param, CI, predictor):
    result, low, high = predictor(param, CI)
    if low > 1:
        low = 1
    if low < 0:
        low = 0
    return low


def get_SL_high(param, CI, predictor):
    result, low, high = predictor(param, CI)
    if high > 1:
        high = 1
    if high < 0:
        high = 0
    return high

header = st.container()
lob_and_freq = st.container()
parameter = st.container()
model_prediction = st.container()
erlangC_prediction = st.container()

with header:
    st.title("Service Level Calculator")
    #st.text("@author: UChicago MScA Capstone Team - Kaicheng Zhang, Alina Zhou, Sherry Zha")
    st.markdown("Use this calculator to determine your expected service level depending on headcount and others.")

with lob_and_freq:
    st.subheader("Select Line of Business and Frequency")
    lob_col, freq_col, days_col = st.columns(3)
    line_of_business_text = lob_col.selectbox('Line of Business', options = ['COMM', 'RESI', 'MYQ'], index = 0)
    if line_of_business_text == 'COMM':
        lob = 0
    elif line_of_business_text == 'RESI':
        lob = 1
    elif line_of_business_text == 'MYQ':
        lob = 2
    else:
        lob = 0
    
    freq_text = freq_col.selectbox('Frequency', options=['Weekly', 'Daily'], index=1)
    if freq_text=='Weekly':
        if lob==0:
            num_of_days = days_col.number_input('Number of Workdays in the Week', min_value=1, max_value=7, value=5, step=1)
        else:
            num_of_days = days_col.number_input('Number of Workdays in the Week', min_value=1, max_value=7, value=6, step=1)

with parameter:
    
    st.subheader('Enter Parameters')
    left_col, mid_col, right_col = st.columns(3)
    
    #catch necesary parameters
    headcount = left_col.number_input(label='Daily Headcount', 
                                        min_value=0,
                                        value = 100,
                                        step=1)
       
    
    
    total_call_volume = left_col.number_input(label='Total Calls Volumes',
                                        min_value=0,
                                        value=2000)

    
    # Now I take average for weekly call volume, will be upgraded later
    if freq_text == 'Weekly':
        call_volume = total_call_volume/num_of_days
        
    else:
        call_volume = total_call_volume
        

    
    aht = mid_col.number_input('Average Handling Time in Seconds',min_value=0, value=700)
    
    apa = mid_col.number_input('Average Speed of Answer in Seconds', min_value = 0, value=100)

    occupancy = right_col.number_input('Maximum Occupancy %', min_value=0.0, max_value=100.0, value=80.0, step=1.0)/100
    
    shrinkage = right_col.number_input('Shrinkage %', min_value=0.0, max_value=100.0, value=30.0, step=1.0)/100
    
    confidence_level = st.slider('Confidence Level %', min_value = 80, max_value = 99, value = 90, step = 1)/100
    
    param = np.array([headcount, call_volume, aht, apa, occupancy, shrinkage])


    # display the results
with model_prediction:
    st.subheader('Machine Learning Results')
    left, mid, right = st.columns(3)
    sl_label = 'Service Level'
    sl_l_label = 'Lower Bound of Confidence Interval'
    sl_h_label = 'Upper Bound of Confidence Interval'
    if lob == 0:
        sl = adjust_percentage(get_SL(param, confidence_level, com_sl_prediction))
        sl_l = adjust_percentage(get_SL_low(param, confidence_level, com_sl_prediction))
        sl_h = adjust_percentage(get_SL_high(param, confidence_level, com_sl_prediction))
        left.metric(label=sl_label, value=str(round(sl*100, 2))+'%')
        mid.metric(label=sl_l_label, value=str(round(sl_l*100, 2))+'%')
        right.metric(label=sl_h_label, value=str(round(sl_h*100, 2))+'%')
    elif lob == 1:
        sl = adjust_percentage(get_SL(param, confidence_level, res_sl_prediction))
        sl_l = adjust_percentage(get_SL_low(param, confidence_level, res_sl_prediction))
        sl_h = adjust_percentage(get_SL_high(param, confidence_level, res_sl_prediction))
        left.metric(label=sl_label, value=str(round(sl*100, 2))+'%')
        mid.metric(label=sl_l_label, value=str(round(sl_l*100, 2))+'%')
        right.metric(label=sl_h_label, value=str(round(sl_h*100, 2))+'%')
    else:
        sl = adjust_percentage(get_SL(param, confidence_level, myq_sl_prediction))
        sl_l = adjust_percentage(get_SL_low(param, confidence_level, myq_sl_prediction))
        sl_h = adjust_percentage(get_SL_high(param, confidence_level, myq_sl_prediction))
        left.metric(label=sl_label, value=str(round(sl*100, 2))+'%')
        mid.metric(label=sl_l_label, value=str(round(sl_l*100, 2))+'%') 
        right.metric(label=sl_h_label, value=str(round(sl_h*100, 2))+'%')
    

with erlangC_prediction:
    st.subheader('Erlang C Results')
    left, mid, right = st.columns(3)
    if freq_text == 'Weekly':
        ec = MyProgram(transactions=total_call_volume, 
                    asa=apa/60, aht=aht/60, interval=60*8*num_of_days, shrinkage=shrinkage, occupancy=occupancy)
    else:
        ec = MyProgram(transactions=total_call_volume, 
                    asa=apa/60, aht=aht/60, interval=60*8, shrinkage=shrinkage, occupancy=occupancy)       
    #left.metric(label='HeadCount', value=ec.required_positions(service_level=service_level)['positions'])

    #mid.write(str(round(ec.required_positions(service_level=service_level)['service_level']*100,2))+'%')
    left.metric(label='Service Level', value=str(round(adjust_percentage(ec.service_level(positions = headcount))*100,2))+'%')
    


    



