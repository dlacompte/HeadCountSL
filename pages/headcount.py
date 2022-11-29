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

# Erlang C class

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

@st.cache
def load_res_hc():
    filename1 = os.path.join(dirname, 'res_hc.joblib')
    model_hc = joblib.load(filename1)
    return model_hc

@st.cache
def load_myq_hc():
    filename2 = os.path.join(dirname, 'myq_hc.joblib')
    model_hc = joblib.load(filename2)
    return model_hc


@st.cache
def load_com_hc():
    filename3 = os.path.join(dirname, 'com_hc.joblib')
    model_hc = joblib.load(filename3)
    return model_hc

#load RMSE
RMSE_com_hc = 2.5656574352948165
RMSE_res_hc = 14.121853781129152
RMSE_myq_hc = 14.727407096077105
columns = ['SL', 'Call Volume','AHT','ASA','Occupancy %','Shrinkage %']

# page config
st.set_page_config(page_title="Headcount Calculator", page_icon="👩‍💻")
st.sidebar.success('Select a Calculator above')
st.sidebar.header('Headcount Calculator')

def res_hc_prediction(param,CI):
    alpha = 1 - CI
    upper = alpha/2
    Z = stat.norm.ppf(1-upper)          #Above three line for Calculating Z-score
    param = pd.DataFrame([param], columns=columns)
    model_hc = load_res_hc()
    result = model_hc.predict(param)[0]    #Prediction of HC
    low = result - Z* RMSE_res_hc                #Upper and Lower bound
    high = result + Z * RMSE_res_hc
    return result, low, high

def com_hc_prediction(param,CI):
    alpha = 1 - CI
    upper = alpha/2
    Z = stat.norm.ppf(1-upper)          #Above three line for Calculating Z-score
    param = pd.DataFrame([param], columns=columns)
    model_hc = load_com_hc()
    result = model_hc.predict(param)[0]    #Prediction of HC
    low = result - Z* RMSE_com_hc                #Upper and Lower bound
    high = result + Z * RMSE_com_hc
    return result, low, high

def myq_hc_prediction(param,CI):
    alpha = 1 - CI
    upper = alpha/2
    Z = stat.norm.ppf(1-upper)          #Above three line for Calculating Z-score
    param = pd.DataFrame([param], columns=columns)
    model_hc = load_myq_hc()
    result = model_hc.predict(param)[0]    #Prediction of HC
    low = result - Z* RMSE_myq_hc                #Upper and Lower bound
    high = result + Z * RMSE_myq_hc
    return result, low, high

def get_HC(param, CI, predictor):
    result, low, high = predictor(param, CI)
    return result


def get_HC_low(param, CI, predictor):
    result, low, high = predictor(param, CI)
    return low

def get_HC_high(param, CI, predictor):
    result, low, high = predictor(param, CI)
    return high



header = st.container()
lob_and_freq = st.container()
parameter = st.container()
model_prediction = st.container()
erlangC_prediction = st.container()

with header:
    st.title("Headcount Calculator")
    st.markdown("Use this calculator to determine your expected headcount (the number of agents) depending on service level and others.")

with lob_and_freq:
    st.subheader("Select Line of Business and Frequency")
    lob_col, freq_col, days_col = st.columns(3)
    line_of_business_text = lob_col.selectbox('Line of Business', options = ['COMM', 'RESI', 'MYQ'], index = 1)
    if line_of_business_text == 'COMM':
        lob = 0
    elif line_of_business_text == 'RESI':
        lob = 1
    elif line_of_business_text == 'MYQ':
        lob = 2
    else:
        lob = 0

    freq_text = freq_col.selectbox('Frequency', options=['Weekly', 'Daily'], index=0)
    if freq_text=='Weekly':
        if lob==0:
            num_of_days = days_col.number_input('Number of Workdays in the Week', min_value=1, max_value=7, value=5, step=1)
        else:
            num_of_days = days_col.number_input('Number of Workdays in the Week', min_value=1, max_value=7, value=6, step=1)

with parameter:

    st.subheader('Enter Parameters')
    left_col, mid_col, right_col = st.columns(3)

    #catch necesary parameters
    service_level = left_col.number_input(label='Service Level Target %',
                                        min_value=0.0,
                                        max_value=100.0,
                                        value = 90.0,
                                        step=1.0)/100



    total_call_volume = left_col.number_input(label='Total Call Volumes',
                                        min_value=0,
                                        value=14800)


    aht = mid_col.number_input('Average Handling Time',min_value=0, value=900,
                                help='Enter your estimated AHT in seconds')

    apa = mid_col.number_input('Average Speed of Answer', min_value = 0, value=50,
                                help='Enter your estimated ASA in seconds, NOT the acceptable wait time')

    occupancy = right_col.number_input('Maximum Occupancy %', min_value=0.0, max_value=100.0, value=65.0, step=1.0)/100

    shrinkage = right_col.number_input('Shrinkage %', min_value=0.0, max_value=100.0, value=40.0, step=1.0)/100

    # Now I take average for weekly call volume, will be upgraded later
    if freq_text == 'Weekly':
        call_volume = total_call_volume/num_of_days
        confidence_level = 0.95  #won't be used

    else:
        call_volume = total_call_volume
        confidence_level = left_col.selectbox('Confidence Level %', options=[90, 95, 99], index=0)/100
        opening_hours = mid_col.number_input('Length of Opening Hours', min_value = 1.0, max_value=24.0, value=13.0, step=1.0,
                                    help='Hours between the opening and closing of the call center; only used in Erlang C results')

    param = np.array([service_level, call_volume, aht, apa, occupancy, shrinkage])

    # display the results
with model_prediction:
    st.subheader('Machine Learning Results')
    left, mid, right = st.columns(3)

    if freq_text == 'Weekly':
        hc_label = 'Avg Daily HeadCount'
        if lob == 0:
            param_min = np.array([service_level, call_volume*0.925, aht, apa, occupancy, shrinkage])
            param_max = np.array([service_level, call_volume*1.084, aht, apa, occupancy, shrinkage])
            avg_hc = max(get_HC(param, confidence_level, com_hc_prediction),0)
            min_hc = max(get_HC(param_min, confidence_level, com_hc_prediction),0)
            max_hc = get_HC(param_max, confidence_level, com_hc_prediction)
            left.metric(label=hc_label, value=round(avg_hc, 1))
            mid.metric(label='Min Daily Headcount in the Week', value=round(min(min_hc, avg_hc), 1))
            right.metric(label='Max Daily Headcount in the Week', value=round(max(max_hc, avg_hc), 1))
        elif lob == 1:
            param_min = np.array([service_level, call_volume*0.542, aht, apa, occupancy, shrinkage])
            param_max = np.array([service_level, call_volume*1.326, aht, apa, occupancy, shrinkage])
            avg_hc = max(get_HC(param, confidence_level, res_hc_prediction),0)
            min_hc = max(get_HC(param_min, confidence_level, res_hc_prediction),0)
            max_hc = get_HC(param_max, confidence_level, res_hc_prediction)
            left.metric(label=hc_label, value=round(avg_hc, 1))
            mid.metric(label='Min Daily Headcount in the Week', value=round(min(min_hc, avg_hc), 1))
            right.metric(label='Max Daily Headcount in the Week', value=round(max(max_hc, avg_hc), 1))
        else:
            param_min = np.array([service_level, call_volume*0.723, aht, apa, occupancy, shrinkage])
            param_max = np.array([service_level, call_volume*1.268, aht, apa, occupancy, shrinkage])
            avg_hc = max(get_HC(param, confidence_level, myq_hc_prediction),0)
            min_hc = max(get_HC(param_min, confidence_level, myq_hc_prediction),0)
            max_hc = get_HC(param_max, confidence_level, myq_hc_prediction)
            left.metric(label=hc_label, value=round(avg_hc, 1))
            mid.metric(label='Min Daily Headcount in the Week', value=round(min(min_hc, avg_hc), 1))
            right.metric(label='Max Daily Headcount in the Week', value=round(max(max_hc, avg_hc), 1))
    else:
        hc_label = 'HeadCount'
        if lob == 0:
            left.metric(label=hc_label, value=round(get_HC(param, confidence_level, com_hc_prediction), 1))
            mid.metric(label='Lower Bound of Confidence Interval', value=round(get_HC_low(param, confidence_level, com_hc_prediction), 1))
            right.metric(label='Upper Bound of Confidence Interval', value=round(get_HC_high(param, confidence_level, com_hc_prediction), 1))
        elif lob == 1:
            left.metric(label=hc_label, value=round(get_HC(param, confidence_level, res_hc_prediction), 1))
            mid.metric(label='Lower Bound of Confidence Interval', value=round(get_HC_low(param, confidence_level, res_hc_prediction), 1))
            right.metric(label='Upper Bound of Confidence Interval', value=round(get_HC_high(param, confidence_level, res_hc_prediction), 1))
        else:
            left.metric(label=hc_label, value=round(get_HC(param, confidence_level, myq_hc_prediction), 1))
            mid.metric(label='Lower Bound of Confidence Interval', value=round(get_HC_low(param, confidence_level, myq_hc_prediction), 1))
            right.metric(label='Upper Bound of Confidence Interval', value=round(get_HC_high(param, confidence_level, myq_hc_prediction), 1))


with erlangC_prediction:
    st.subheader('Erlang C Results')
    left, mid, right = st.columns(3)
    if freq_text == 'Weekly':
        ec = MyProgram(transactions=total_call_volume,
                    asa=apa/60, aht=aht/60, interval=60*13*num_of_days, shrinkage=shrinkage, occupancy=occupancy)
    else:
        ec = MyProgram(transactions=total_call_volume,
                    asa=apa/60, aht=aht/60, interval=60*opening_hours, shrinkage=shrinkage, occupancy=occupancy)
    left.metric(label='HeadCount', value=ec.required_positions(service_level=service_level)['positions'])

    #mid.write(str(round(ec.required_positions(service_level=service_level)['service_level']*100,2))+'%')
    mid.metric(label='Service Level', value=str(round(ec.required_positions(service_level=service_level)['service_level']*100,2))+'%')
