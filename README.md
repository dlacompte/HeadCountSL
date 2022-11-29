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
