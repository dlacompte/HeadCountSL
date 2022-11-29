import warnings

import numpy as np
import pandas as pd
import scipy.stats as stat
# Model
import sklearn
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
# io
import os
import io
import joblib
import pickle
import zipfile
warnings.filterwarnings("ignore")

RMSE_com_sl = 0.046036637305982606
RMSE_myq_sl = 0.04836831916059791
RMSE_res_sl = 0.07258282171863702

RMSE_com_hc = 2.5656574352948165
RMSE_res_hc = 14.121853781129152
RMSE_myq_hc = 14.727407096077105

upload_csv = st.container()
data_cleaning = st.container()
fine_tunning = st.container()

with upload_csv:
    st.header('Upload your cleaned data spreadsheet')
    st.markdown('Make sure the spreadsheet contains data in **only one line of business**.')
    line_of_business_text = st.selectbox('What Line of Business are you training for?', options = ['COM', 'RES', 'MYQ'], index = 0)

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
    st.header('Check Data...')
    df_selected = df.dropna()
    rows = len(df_selected)
    st.write("You have ", rows, " rows of valid observations after cleaning.")

    df_show = st.expander("Check the statistics of your data")
    df_show.dataframe(df_selected.describe())
    df_selected = df_selected[['HC','SL', 'Call Volume','AHT','ASA','Occupancy %','Shrinkage %']]
    if line_of_business_text == 'COM':
        hc_rmse = RMSE_com_hc
        sl_rmse = RMSE_com_sl
    elif line_of_business_text == 'RES':
        hc_rmse = RMSE_res_hc
        sl_rmse = RMSE_res_sl
    else:
        hc_rmse = RMSE_myq_hc
        sl_rmse = RMSE_myq_sl

with fine_tunning:
    st.header('Modeling...')
    # train_test split
    df_train, df_test = train_test_split(df_selected, test_size=0.2, random_state=2022)

    # HC prediction
    X_train_hc = df_train.drop(columns='HC')  #predictors, all other variables other than "headcount"
    y_train_hc = df_train[['HC']]      #response variable, 'Headcount'

    X_test_hc = df_test.drop(columns=['HC'])  #predictors, all other variables other than "headcount"
    y_test_hc = df_test[['HC']]      #response variable, 'Headcount'


    # preprocessing
    norm_col = ['Call Volume','AHT', 'Occupancy %','Shrinkage %']
    skew_col = ['ASA','SL']

    ss=ColumnTransformer([
        ('norm',StandardScaler(),norm_col),
        ('skew',PowerTransformer(method='box-cox'),skew_col)
        ])

    # grid search on Gradiant Boosting model

    k_fold = KFold(n_splits=5, random_state=0, shuffle=True)
    reg = GradientBoostingRegressor()
    pipe_hc = Pipeline([('scaler', ss), ('reg', reg)])

    param1 = {
        'reg__min_samples_split':[3,5,10,15,20,25,30,40,50],
        'reg__max_leaf_nodes':[3,5,10,15,20,25,30,40,50]
    }
    gs = GridSearchCV(estimator=pipe_hc, param_grid=param1, cv=k_fold, scoring='neg_mean_squared_error')

    # SL Prediction
    X_train_sl = df_train.drop(columns='SL')  #predictors, all other variables other than "service_level"
    y_train_sl = df_train[['SL']]      #response variable, 'service_level'

    X_test_sl = df_test.drop(columns=['SL'])  #predictors, all other variables other than "service_level"
    y_test_sl = df_test[['SL']]      #response variable, 'service_level'

    # preprocessing
    norm_col2 = ['Call Volume','AHT', 'Occupancy %','Shrinkage %', 'HC']
    skew_col2 = ['ASA']

    ss2=ColumnTransformer([
        ('norm',StandardScaler(),norm_col2),
        ('skew',PowerTransformer(method='box-cox'),skew_col2)
        ])

    # random search on Random Forest models

    reg2 = RandomForestRegressor()

    param2 = {'reg__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'reg__min_samples_leaf': [1, 2, 4],
            'reg__min_samples_split': [2, 5, 10],
            'reg__n_estimators': [100, 200, 400, 600, 800]}
    pipe_sl = Pipeline([('scaler', ss2), ('reg', reg2)])

    rs = RandomizedSearchCV(estimator=pipe_sl, param_distributions=param2, cv = k_fold, scoring='neg_mean_squared_error')


    # click on the button to train models

    if st.button('Start Modeling'):
        st.subheader('Modeling HeadCount')
        with st.spinner('Modeling... It will take a few minutes...'):
            gs.fit(X_train_hc, y_train_hc)
            st.success('Headcount Done!')
            #st.write(gs.best_params_)
            st.markdown("**Root Mean Squared Error (RMSE) Comparison**")
            left, mid, right = st.columns(3)
            left.metric(label='RMSE of train set', value=round(np.sqrt(-gs.best_score_),2))
            pipe_final = gs.best_estimator_
            y_pred = pipe_final.predict(X_test_hc)
            mid.metric(label='RMSE of test set', value=round(np.sqrt(mean_squared_error(y_test_hc, y_pred)),2))
            right.metric(label='RMSE of original model', value=round(hc_rmse,2))
        st.subheader('Modeling Service Level')
        with st.spinner('Modeling... It will take a few minutes...'):
            rs.fit(X_train_sl, y_train_sl)
            st.success('Service Level Done!')
            st.markdown("**Root Mean Squared Error (RMSE) Comparison**")
            left, mid, right = st.columns(3)
            left.metric(label='RMSE of train set', value=str(round(np.sqrt(-rs.best_score_)*100,2))+'%')
            pipe_final_sl = rs.best_estimator_
            y_pred = pipe_final_sl.predict(X_test_sl)
            mid.metric(label='RMSE of test set', value=str(round(np.sqrt(mean_squared_error(y_test_sl, y_pred))*100,2))+'%')
            right.metric(label='RMSE of original model', value=str(round(sl_rmse*100,2))+'%')
    else:
        st.stop()

    st.info('''If you are satisfied with the new models, click on 'Download Model' to download.
                If you want to retrain models, upload a new spreadsheet above to start over.''')

    # fit the final pipelines
    pipe_final.fit(X_train_hc, y_train_hc)
    pipe_final_sl.fit(X_train_sl, y_train_sl)

    # save the pipelines
    hc_model = pickle.dumps(pipe_final, protocol=pickle.HIGHEST_PROTOCOL )
    sl_model = pickle.dumps(pipe_final_sl, protocol=pickle.HIGHEST_PROTOCOL )

    # zip two pipelines
    zip_model = io.BytesIO()
    with zipfile.ZipFile(file=zip_model, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(zinfo_or_arcname=line_of_business_text.lower()+'_hc.joblib',data=hc_model)
        z.writestr(zinfo_or_arcname=line_of_business_text.lower()+'_sl.joblib',data=sl_model)

    zip_final = zip_model.getvalue()
    zip_model.close()

    # download the zip file
    st.download_button(
        "Download Model",
        data=zip_final,
        file_name="new_models.zip",
    )
