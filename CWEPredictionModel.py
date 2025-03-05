import pandas as pd
import numpy as np
import requests
import json
import os
import re
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

# 1. Data Collection
def fetch_nvd_data(year, api_key=None):
    """
    Fetch CVE data from NVD for a specific year
    """
    base_url = f"https://services.nvd.nist.gov/rest/json/cves/2.0"
    
    # Parameters for the API request
    params = {
        'pubStartDate': f"{year}-01-01T00:00:00.000",
        'pubEndDate': f"{year}-03-31T23:59:59.999",
        'resultsPerPage': 2000  # Maximum allowed
    }
    # https://services.nvd.nist.gov/rest/json/cves/2.0/?pubStartDate=2021-08-04T00:00:00.000&pubEndDate=2021-10-22T00:00:00.000

    # Add API key if provided (recommended for production use)
    headers = {}
    if api_key:
        headers['apiKey'] = api_key
    
    print(f"Fetching CVE data for {year}...")
    
    # Initial request
    response = requests.get(base_url, params=params, headers=headers)
    
    if response.status_code != 200:
        print(f"request:  {response}")
        print(f"Error fetching data: {response.status_code}")
        return None
    
    data = response.json()
    total_results = data.get('totalResults', 0)
    results = data.get('vulnerabilities', [])
    
    print(f"Found {total_results} total vulnerabilities")
    
    # If there are more results, use pagination to get them all
    start_index = 2000
    while len(results) < total_results and start_index < 10000:  # Limit to 10,000 for demo
        params['startIndex'] = start_index
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code == 200:
            batch = response.json().get('vulnerabilities', [])
            if not batch:
                break
            results.extend(batch)
            start_index += len(batch)
            print(f"Retrieved {len(results)} of {total_results}")
        else:
            print(f"Error in pagination: {response.status_code}")
            break
    
    return results

def extract_cve_cwe_pairs(vulnerabilities):
    """
    Extract CVE descriptions and associated CWE IDs
    """
    cve_cwe_pairs = []
    
    for vuln in vulnerabilities:
        cve_item = vuln.get('cve', {})
        cve_id = cve_item.get('id')
        
        # Get description
        descriptions = cve_item.get('descriptions', [])
        english_desc = next((d['value'] for d in descriptions if d.get('lang') == 'en'), None)
        
        if not english_desc:
            continue
        
        # Get CWE ID(s)
        weaknesses = cve_item.get('weaknesses', [])
        cwe_ids = []
        
        for weakness in weaknesses:
            for desc in weakness.get('description', []):
                if desc.get('lang') == 'en' and 'CWE-' in desc.get('value', ''):
                    # Extract CWE ID(s) from the description
                    matches = re.findall(r'CWE-\d+', desc.get('value', ''))
                    cwe_ids.extend(matches)
        
        # Only keep entries that have both description and at least one CWE
        if english_desc and cwe_ids:
            cve_cwe_pairs.append({
                'cve_id': cve_id,
                'description': english_desc,
                'cwe_ids': cwe_ids
            })
    
    return cve_cwe_pairs

# 2. Preprocess the data
def preprocess_text(text):
    """
    Basic text preprocessing
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keeping only letters, numbers and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 3. Prepare the data for training
def prepare_data(cve_cwe_pairs, top_n_cwes=20):
    """
    Prepare the data for model training:
    - Keep only top N most common CWE categories
    - Create dataframe with one-hot encoded CWE labels
    """
    # Extract all CWE IDs and count frequencies
    all_cwe_ids = []
    for item in cve_cwe_pairs:
        all_cwe_ids.extend(item['cwe_ids'])
    
    cwe_counter = Counter(all_cwe_ids)
    top_cwes = [cwe for cwe, _ in cwe_counter.most_common(top_n_cwes)]
    
    # Create DataFrame
    data = []
    for item in cve_cwe_pairs:
        # Keep only entries that have at least one of the top CWEs
        item_cwes = set(item['cwe_ids'])
        if not item_cwes.intersection(top_cwes):
            continue
        
        # Preprocess description
        preprocessed_desc = preprocess_text(item['description'])
        
        # One-hot encode CWE labels
        cwe_labels = {cwe: 1 if cwe in item['cwe_ids'] else 0 for cwe in top_cwes}
        
        entry = {
            'cve_id': item['cve_id'],
            'description': preprocessed_desc,
            **cwe_labels
        }
        
        data.append(entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df, top_cwes

# 4. Train a model
def train_model(df, top_cwes):
    """
    Train a multi-label classification model
    """
    # Split into features and target
    X = df['description']
    y = df[top_cwes]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train a multi-label classifier
    classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_tfidf)
    
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=top_cwes))
    
    # Plot CWE distribution
    plt.figure(figsize=(12, 6))
    y_train.sum().sort_values(ascending=False).plot(kind='bar')
    plt.title('CWE Distribution in Training Data')
    plt.xlabel('CWE ID')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('cwe_distribution.png')
    
    return vectorizer, classifier

# 5. Create a prediction function
def predict_cwe(description, vectorizer, classifier, top_cwes, threshold=0.3):
    """
    Predict CWE categories for a given CVE description
    """
    # Preprocess the input
    preprocessed = preprocess_text(description)
    
    # Vectorize
    description_tfidf = vectorizer.transform([preprocessed])
    
    # Predict probabilities
    proba = classifier.predict_proba(description_tfidf)
    
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

# Main execution flow
def main(years=None, api_key=None):
    if years is None:
        years = [2023]  # Default to just last year for demo
    
    all_vulnerabilities = []
    for year in years:
        vulnerabilities = fetch_nvd_data(year, api_key)
        if vulnerabilities:
            all_vulnerabilities.extend(vulnerabilities)
    
    if not all_vulnerabilities:
        print("No data fetched. Exiting.")
        return
    
    print(f"Processing {len(all_vulnerabilities)} vulnerabilities...")
    cve_cwe_pairs = extract_cve_cwe_pairs(all_vulnerabilities)
    print(f"Extracted {len(cve_cwe_pairs)} CVE-CWE pairs")
    
    df, top_cwes = prepare_data(cve_cwe_pairs)
    print(f"Prepared data with {len(df)} entries and {len(top_cwes)} top CWEs")
    
    vectorizer, classifier = train_model(df, top_cwes)
    
    # Save the trained model and vectorizer
    import pickle
    with open('cve_cwe_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open('cve_cwe_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    with open('top_cwes.pkl', 'wb') as f:
        pickle.dump(top_cwes, f)
    
    print("Model saved. Testing with a sample description...")
    
    # Test with a sample description
    sample_description = """
    A vulnerability in the web interface of Product X allows remote attackers 
    to execute arbitrary code via a crafted HTTP request. The issue occurs due 
    to improper input validation in the login form.
    """
    
    predictions = predict_cwe(sample_description, vectorizer, classifier, top_cwes)
    
    print("\nPredicted CWEs for sample description:")
    for pred in predictions:
        print(f"{pred['cwe']}: {pred['probability']:.4f}")

# Run the demo if executed directly
if __name__ == "__main__":
    # Replace with your API key if you have one
    # api_key = "your-api-key-here"
    load_dotenv()
    api_key = os.getenv('NVD_API_KEY')
    
    # Specify years to fetch data for
    years = [2022, 2023]
    
    main(years, api_key)

# Example of loading and using a saved model
def load_and_predict(description):
    import pickle
    
    # Load the model components
    with open('cve_cwe_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('cve_cwe_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    
    with open('top_cwes.pkl', 'rb') as f:
        top_cwes = pickle.load(f)
    
    # Make a prediction
    predictions = predict_cwe(description, vectorizer, classifier, top_cwes)
    
    return predictions