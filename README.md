# uchicago-capstone
Chamberlain Group Call Center headcount and service level calculator  
Click here to get started! https://chamberlain-calculator.streamlitapp.com/ 

## Introduction to Files
### /uchicago-capstone main directory
* welcome.py : the homepage of the application. Make sure Streamlit links to this main file.  
* requirements.txt: a file listing all the dependencies for a specific Python project. It does not need any changes unless you add any new packages to the source code.  
* chamberlain-logo.png and university_of_chicago.png: just logos used in the homepage.
### /pages subdirectory
Note: Do not move or rename any file under this directory.
* headcount.py: headcount calculator page
* service_level.py: service level calculator page
* train_new_model.py: train new model page
* files ended with '.joblib': machine learning models for calculation. e.g. 'com_hc.joblib' is responsible for headcount predictions of the commercial line.
* model_data_template.xlsx: a spreadsheet template to contain new data to train new models

## Model Training Instruction 
The "train new model" page provides a way to build new machine learning models based on new data. You can download the models and replace the previous ones in the Github. 
### When to train new models
* There is a huge expansion in call volume or headcounts, and the current models cannot give a reasonable result.
* We have more than 1000 rows of valid daily data for a line of business.
### Steps to train new models
* Download the template "model_data_template.xlsx" from "page" folder; fill in the data into the template according to the notes inside; make sure there is only data from one line of business in the spreadsheet.
* Drop any outliers or anomalies in the spreadsheet.
* Upload the cleaned spreadsheet to the webpage; select the LOB
* Check your uploaded data and the statistics; if it looks good, you can do the modeling, or you need to edit the spreadsheet and upload it again.
* Click on "Start modeling"; it will train the models for headcount and service levels automatically.
* Compare the root mean squared errors of train set, test set, and the original model; if the following conditions are met, you may consider using new models:
    1. train set RMSE ≈ test set RMSE
    2. test set RMSE <= original RMSE
* Once you decide to replace the models, click on the "Download models" button to download a zipfile containing both headcount and service level models.
* Upload your new models to github "page" directory; click on "Commit changes", and github will overwrite the old one.
