import warnings

import numpy as np
import pandas as pd
import scipy.stats as stat
# Model
import sklearn
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib
import pickle
warnings.filterwarnings("ignore")


upload_csv = st.container()
data_cleaning = st.container()
fine_tunning = st.container()

with upload_csv:
    st.header('Upload an Excel sheet')
    uploaded_file = st.file_uploader(
        label = 'Upload',
        label_visibility='hidden',
        key="1",
        #help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

    if uploaded_file is not None:
        file_container = st.expander("Check your uploaded .xlsx")
        df = pd.read_excel(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(df)

    else:
        st.info(
            f"""
                👆 Upload a .xlsx file first. 
                """
        )

        st.stop()

with data_cleaning:
    st.header('Data Cleaning...')
    line_of_business_text = st.selectbox('Select a Line of Business', options = ['COM', 'RES', 'MYQ'], index = 0)
    df_selected = df.loc[df.LOB == line_of_business_text].drop(columns='LOB')
    df_show = st.expander("Check your selected data")
    df_show.dataframe(df_selected)
    df_selected = df_selected[['HC','SL', 'Call Volume','AHT','ASA','Occupancy %','Shrinkage %']]
    st.write('There will be some filters to clean the data...')

with fine_tunning:
    st.header('Modeling...')
    # train_test split
    df_train, df_test = train_test_split(df_selected, test_size=0.1, random_state=2022) 

    # HC prediction
    X_train_hc = df_train.drop(columns='HC')  #predictors, all other variables other than "headcount"
    y_train_hc = df_train[['HC']]      #response variable, 'Headcount'

    X_test_hc = df_test.drop(columns=['HC'])  #predictors, all other variables other than "headcount"
    y_test_hc = df_test[['HC']]      #response variable, 'Headcount'


    # preprocessing
    ss=StandardScaler()
    ss.fit(X_train_hc)
    X_train_hc = ss.transform(X_train_hc) 

    # grid search on Gradiant Boosting model

    k_fold = KFold(n_splits=10, random_state=0, shuffle=True)
    reg = GradientBoostingRegressor()
    param1 = {
        'min_samples_split':[3,5,10,15,20,25,30,40,50],
        'max_leaf_nodes':[3,5,10,15,20,25,30,40,50]
    }
    gs = GridSearchCV(estimator=reg, param_grid=param1, cv=k_fold, scoring='neg_mean_squared_error')

    if st.button('Start Modeling'):
        with st.spinner('Wait for it...'):
            gs.fit(X_train_hc, y_train_hc) 
            st.success('Done!')
            st.write(gs.best_params_)   
            st.write('Root Mean Squared Error: '+ str(np.sqrt(-gs.best_score_)))
    else:
        st.stop()

    
    pipe_hc = Pipeline([('scaler', ss), ('reg', gs.best_params_)])
    import os

    dirname = os.path.dirname(__file__)
    filename1 = os.path.join(dirname, 'hc_test.pkl')
    joblib.dump(pipe_hc, filename1)
    st.download_button(
        "Download Model",
        data=pickle.dumps(pipe_hc),
        file_name="hc_test.pkl",
    )

    


