# Credit Risk Classification Assessment;
## Project Overview

This document is a documentation of the processes and steps taken in addressing this credit risk classification problem. The task at hand entails using the [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) to calculate probability of a borrower defaulting on their loan. An a simple interface is then created to display the probability of a borrower defaulting on their loan payment. 

We use a simple logistical regression model to determine the probability of default, using the data given. For our model, three key features used from the dataset: Debt-to-income ratio, revolving utilisation rate, and the lower boundary range the borrower’s last FICO pulled belongs to (credit score).  


## Data
### Data Preparation

The following steps were taken in preparing the data to be used in this exercise:

 + The data was downloaded and stored onto the local drive. The data was saved as a csv files.
 + Two datasets were downloaded. One containing records of accepted loans [accepted_2007_to_2018Q4.csv], 
 and one containing records of rejected loans [rejected_2007_to_2018Q4.csv].
 Because we are dealing we are trying to get the probability of credit holders defaulting on payments, only the accepted loans data was
 this exercise. 
 + The downloaded data is stored in the [dataset] folder 
 + The data dictionary was also downloaded [LCDataDictionary.xlsx]. This was saved as an xlsx file .
 + The data dictionary was used to understand the headings of the downloaded data.
 
 

 ### Data Exploration and Cleaning
 The data was imported into Python for exploration. The following data discoveries were made:

 + The data contained over 2,260,701 records (rows) and  around 151 variables (columns).
 + A few features had datapoints of mixed data types. However, for the type of modelling that was required in the exercise, 
 the most important datapoint that required cleaning was the assigned ID for loan listings ('id'). 
 + The 'id' variable had integer and string datapoints.
 + The other variables (e.g NaN 'next_pymnt_d', 'desc') had a mixture of NaN datapoints (float) and string or date dapaoints  in their string  


## Backend

We first start by importing all the necessary libraries into the program. These include:

+ Flask - for 
+ CORS -
+ pandas - 
+ numpy - 
+ LogisticRegression - 
+ logging -  

We start by first configuring logging process of the program.
We want it to display logs that give information at the INFO logging level.

We then create a Flask web application. We allow the web application to be accessed by other applications
through the Cross-Origin Resource Sharing CORS.


We then create a function that will initialise our model for calculating the probability of default.


 ### Guide on how to run the backend
 + Step 1 Open VS Code from the Credit Risk Assessment folder. 
 + Step 2 Once in this folder open the [Terminal].
 + Step 3 In Powershell, run the `cd backend` command to go into the [backend] directory. Press Enter
 + Step 4 Once in the [backend] directory, run the following command in powershell: `python app.py`. This will initialise the app.py script in backend. 
 + Step 5 Once the app ahs been successfully initialised, the terminal will show the following message with a URL link: [Running on http://127.0.0.1:8000]. Press Cntr and click on the URL link.
 
 + Step 6 Go back to the terminal. Open a new terminal through `Ctrl`+`Shift`+``\`.
 + Step 7 In powershell, run the `cd frontend` command to go into the [frontend] directory.
 + Step 8 Once in the [backend] directory, run the following command in powershell: `streamlit run streamlit_app.py`. This will load a web application.  
 +  Step 9 Once the application has been loaded, the probability of default of a client can now be calculated by simply typing in their Lending Club 'id' number in the text box on display.   


