from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables to store our model
model = None

def initialize_model():
    global model
    
    try:
        # Load and prepare data
        df_accept = pd.read_csv(r'c:\Users\mafaesami\.cache\kagglehub\datasets\wordsforthewise\lending-club\versions\3\accepted_2007_to_2018q4\accepted_2007_to_2018Q4.csv')
        #df_accept = pd.read_csv(r'c:\Users\mafaesami\.cache\kagglehub\datasets\wordsforthewise\lending-club\versions\3\accepted_2007_to_2018q4\accepted_2007_to_2018Q4_reduced.csv')
        #df_accept = df_accept(250000)

        
        # Clean and prepare data
        df_accept['id'] = df_accept['id'].apply(lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else None)
        df_accept = df_accept.dropna(subset=['id'])
        
               
        # Setting default values
        df_accept['default'] = np.nan
        default_0_conditions = [
            'Fully Paid', 'Current', 'Late (31-120 days)', 
            'Late (16-30 days)', 'Does not meet the credit policy. Status:Fully Paid'
        ]
        default_1_conditions = [
            'Charged Off', 'In Grace Period', 'Default',
            'Does not meet the credit policy. Status:Charged Off'
        ]
        
        for condition in default_0_conditions:
            df_accept.loc[df_accept['loan_status'] == condition, 'default'] = 0
        for condition in default_1_conditions:
            df_accept.loc[df_accept['loan_status'] == condition, 'default'] = 1
        
        #Data cleaning - Removing null values for on features 
        df_accept = df_accept[pd.notna(df_accept['dti'])]
        df_accept = df_accept[pd.notna(df_accept['revol_util'])]
        
        #Prepare features for modelling
        X = df_accept[['last_fico_range_low', 'dti', 'revol_util']]
        y = df_accept['default']
        
        # Train the model
        model = LogisticRegression()
        model.fit(X, y)
        
        logger.info("Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is initialized
        if model is None:
            logger.error("Model not initialized")
            return jsonify({'error': 'Model not initialized'}), 500
            
        # Log incoming request
        logger.info("Received prediction request")
        
        # Get data from request
        data = request.get_json()
        logger.info(f"Input data: {data}")
        
        # Validate required fields
        required_fields = ['last_fico_range_low', 'dti', 'revol_util']
        for field in required_fields:
            if field not in data:
                error_msg = f"Missing required field: {field}"
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
        
        # Create DataFrame from input data
        try:
            input_data = pd.DataFrame({
                'last_fico_range_low': [float(data['last_fico_range_low'])],
                'dti': [float(data['dti'])],
                'revol_util': [float(data['revol_util'])]
            })
        except ValueError as e:
            error_msg = f"Invalid data format: {str(e)}"
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
            
        # Make prediction
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        response = {
            'default_probability': float(probability),
            'prediction': int(prediction)
        }
        logger.info(f"Prediction result: {response}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_by_id', methods=['POST'])
def predict_by_id():
    try:
        # Check if model is initialized
        if model is None:
            logger.error("Model not initialized")
            return jsonify({'error': 'Model not initialized'}), 500
            
        # Get ID from request
        data = request.get_json()
        if 'id' not in data:
            return jsonify({'error': 'ID not provided'}), 400
            
        search_id = data['id']
        
        # Load and find the record
        df = pd.read_csv(r'C:\Users\mafaesami\.cache\kagglehub\datasets\wordsforthewise\lending-club\versions\3\accepted_2007_to_2018q4\accepted_2007_to_2018Q4.csv')
        #df = pd.read_csv(r'C:\Users\mafaesami\.cache\kagglehub\datasets\wordsforthewise\lending-club\versions\3\accepted_2007_to_2018q4\accepted_2007_to_2018Q4_reduced.csv')
        record = df[df['id'] == search_id]
        
        if record.empty:
            return jsonify({'error': 'ID not found'}), 404
            
        # Extract features for prediction
        input_data = record[['last_fico_range_low', 'dti', 'revol_util']]
        
        # Make prediction
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        response = {
            'id': search_id,
            'default_probability': float(probability),
            'prediction': int(prediction)
        }
        logger.info(f"Prediction result for ID {search_id}: {response}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction by ID: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if initialize_model():
        app.run(host='127.0.0.1', port=8000, debug=True)
    else:
        logger.error("Failed to initialize model. Exiting.") 