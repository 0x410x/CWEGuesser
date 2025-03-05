# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Text preprocessing function
def preprocess_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Global variables to store model components
classifier = None
top_cwes = None
embedding_model = None
model_loaded = False

# Load the trained model components
def load_model():
    global classifier, top_cwes, embedding_model, model_loaded
    
    try:
        # Load classifier and top CWEs
        with open('bert_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        
        with open('top_cwes.pkl', 'rb') as f:
            top_cwes = pickle.load(f)
        
        with open('bert_model_name.pkl', 'rb') as f:
            bert_model_name = pickle.load(f)
        
        # Load BERT model
        embedding_model = SentenceTransformer(bert_model_name)
        
        model_loaded = True
        print(f"Model loaded successfully. Using BERT model: {bert_model_name}")
        return True
        
    except FileNotFoundError as e:
        model_loaded = False
        print(f"Error loading model: {e}")
        return False

# Load model when the application starts
with app.app_context():
    load_model()

# Prediction function
def predict_cwe_with_bert(description, threshold=0.3):
    global classifier, top_cwes, embedding_model
    
    # Preprocess the input
    preprocessed = preprocess_text(description)
    
    # Generate embedding
    embedding = embedding_model.encode([preprocessed])[0].reshape(1, -1)
    
    # Predict probabilities
    proba = classifier.predict_proba(embedding)
    
    # Get predictions above threshold
    predictions = []
    for i, cwe in enumerate(top_cwes):
        if proba[0][i] >= threshold:
            predictions.append({
                'cwe': cwe,
                'probability': float(proba[0][i])
            })
    
    # Sort by probability
    predictions = sorted(predictions, key=lambda x: x['probability'], reverse=True)
    
    return predictions

# Home page
@app.route('/')
def home():
    global model_loaded
    return render_template('index.html', model_loaded=model_loaded)

# Prediction API endpoint (for AJAX calls or direct API use)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    global model_loaded
    
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please ensure model files exist.',
            'predictions': []
        }), 503
    
    # Get description from request
    data = request.get_json()
    description = data.get('description', '')
    
    if not description:
        return jsonify({
            'error': 'No description provided',
            'predictions': []
        }), 400
    
    # Make prediction
    try:
        predictions = predict_cwe_with_bert(description)
        
        # Return as JSON
        return jsonify({
            'description': description,
            'predictions': predictions,
            'model_type': 'BERT'
        })
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'predictions': []
        }), 500

# Web form submission endpoint
@app.route('/predict', methods=['POST'])
def web_predict():
    global model_loaded
    
    if not model_loaded:
        return render_template('index.html', 
                              error='Model not loaded. Please ensure model files exist.',
                              model_loaded=False)
    
    # Get description from form
    description = request.form.get('description', '')
    
    if not description:
        return render_template('index.html', 
                              error='Please enter a vulnerability description.',
                              model_loaded=True)
    
    # Make prediction
    try:
        predictions = predict_cwe_with_bert(description)
        
        # Return results page
        return render_template('results.html', 
                              description=description, 
                              predictions=predictions,
                              model_loaded=True,
                              model_type='BERT')
    except Exception as e:
        return render_template('index.html', 
                              error=f'Prediction error: {str(e)}',
                              model_loaded=True)

# CWE information endpoints (to get more details about a particular CWE)
@app.route('/cwe/<cwe_id>')
def cwe_info(cwe_id):
    # This could be expanded to fetch information from a CWE database
    # For now, we'll return basic information
    cwe_descriptions = {
        'CWE-79': 'Improper Neutralization of Input During Web Page Generation (Cross-site Scripting)',
        'CWE-787': 'Out-of-bounds Write',
        'CWE-89': 'Improper Neutralization of Special Elements used in an SQL Command (SQL Injection)',
        'CWE-125': 'Out-of-bounds Read',
        'CWE-20': 'Improper Input Validation',
        'CWE-416': 'Use After Free',
        'CWE-22': 'Improper Limitation of a Pathname to a Restricted Directory (Path Traversal)',
        'CWE-352': 'Cross-Site Request Forgery (CSRF)',
        'CWE-78': 'Improper Neutralization of Special Elements used in an OS Command (OS Command Injection)',
        'CWE-862': 'Missing Authorization',
        'CWE-120': 'Buffer Copy without Checking Size of Input (Classic Buffer Overflow)',
        'CWE-200': 'Exposure of Sensitive Information to an Unauthorized Actor',
        'CWE-476': 'NULL Pointer Dereference',
        'CWE-77': 'Improper Neutralization of Special Elements used in a Command (Command Injection)',
        'CWE-434': 'Unrestricted Upload of File with Dangerous Type',
        'CWE-119': 'Improper Restriction of Operations within the Bounds of a Memory Buffer',
        'CWE-121': 'Stack-based Buffer Overflow',
        'CWE-287': 'Improper Authentication',
        'CWE-190': 'Integer Overflow or Wraparound',
        'CWE-863': 'Incorrect Authorization',
    }
    
    description = cwe_descriptions.get(cwe_id, f'Information for {cwe_id} not available')
    
    return jsonify({
        'cwe_id': cwe_id,
        'description': description,
        'url': f'https://cwe.mitre.org/data/definitions/{cwe_id[4:]}.html'
    })

if __name__ == '__main__':
    app.run(debug=True)