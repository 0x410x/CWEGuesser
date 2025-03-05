import pandas as pd
import numpy as np
import json
import re
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# BERT-specific imports
from sentence_transformers import SentenceTransformer
import torch

# 1. Data Loading from Local File (same as before)
def load_nvd_data(file_path):
    """
    Load CVE data from a local NVD JSON file
    """
    print(f"Loading CVE data from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Check the format of the JSON file
        if 'CVE_Items' in data:
            # NVD JSON 1.1 format
            vulnerabilities = data.get('CVE_Items', [])
            print(f"Loaded {len(vulnerabilities)} vulnerabilities in NVD JSON 1.1 format")
            return vulnerabilities, '1.1'
        elif 'vulnerabilities' in data:
            # NVD JSON 2.0 format
            vulnerabilities = data.get('vulnerabilities', [])
            print(f"Loaded {len(vulnerabilities)} vulnerabilities in NVD JSON 2.0 format")
            return vulnerabilities, '2.0'
        else:
            print("Unknown JSON format")
            return [], None
    except Exception as e:
        print(f"Error loading file: {e}")
        return [], None

# Extract CVE-CWE pairs (same as before)
def extract_cve_cwe_pairs(vulnerabilities, format_version):
    """
    Extract CVE descriptions and associated CWE IDs
    """
    cve_cwe_pairs = []
    
    if format_version == '1.1':
        # Process NVD JSON 1.1 format
        for vuln in vulnerabilities:
            cve_item = vuln.get('cve', {})
            cve_id = cve_item.get('CVE_data_meta', {}).get('ID')
            
            # Get description
            descriptions = cve_item.get('description', {}).get('description_data', [])
            english_desc = next((d['value'] for d in descriptions if d.get('lang') == 'en'), None)
            
            if not english_desc:
                continue
            
            # Get CWE ID(s)
            problemtype_data = cve_item.get('problemtype', {}).get('problemtype_data', [])
            cwe_ids = []
            
            for problemtype in problemtype_data:
                for desc in problemtype.get('description', []):
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
    
    elif format_version == '2.0':
        # Process NVD JSON 2.0 format
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

# 2. Text Preprocessing - simplified for BERT
def preprocess_text(text):
    """
    Basic text preprocessing for BERT
    - BERT handles casing, so we don't lowercase
    - BERT uses wordpiece tokenization, so we don't need to remove special characters
    - We just clean up extra whitespace
    """
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
    
    # Print the top CWEs and their counts
    print("\nTop CWE categories:")
    for cwe, count in cwe_counter.most_common(top_n_cwes):
        print(f"{cwe}: {count} occurrences")
    
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

# 4. Generate BERT embeddings
def generate_bert_embeddings(texts, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    Generate BERT embeddings for a list of texts using SentenceTransformers
    """
    print(f"Loading BERT model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Generating BERT embeddings for {len(texts)} texts (in batches of {batch_size})...")
    
    # Process in batches to avoid memory issues
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
        all_embeddings.append(batch_embeddings)
        
        # Print progress
        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(texts):
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
    
    # Combine all batches
    embeddings = np.vstack(all_embeddings)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

# 5. Train a model with BERT embeddings
def train_model_with_bert(df, top_cwes, bert_model_name='all-MiniLM-L6-v2'):
    """
    Train a multi-label classification model using BERT embeddings
    """
    # Split into features and target
    X_texts = df['description'].values
    y = df[top_cwes].values
    
    # Split into train and test sets
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(X_texts, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining on {len(X_train_texts)} examples, testing on {len(X_test_texts)} examples")
    
    # Generate BERT embeddings
    print("Generating embeddings for training set...")
    X_train_embeddings = generate_bert_embeddings(X_train_texts, model_name=bert_model_name)
    
    print("Generating embeddings for test set...")
    X_test_embeddings = generate_bert_embeddings(X_test_texts, model_name=bert_model_name)
    
    # Train a multi-label classifier
    print("Training classifier...")
    classifier = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced'))
    classifier.fit(X_train_embeddings, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_embeddings)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=top_cwes))
    
    # Plot CWE distribution
    plt.figure(figsize=(12, 6))
    pd.Series(y_train.sum(axis=0), index=top_cwes).sort_values(ascending=False).plot(kind='bar')
    plt.title('CWE Distribution in Training Data')
    plt.xlabel('CWE ID')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('cwe_distribution.png')
    print("Saved CWE distribution plot to 'cwe_distribution.png'")
    
    # Create a reusable embedding model
    embedding_model = SentenceTransformer(bert_model_name)
    
    return embedding_model, classifier

# 6. Create a prediction function
def predict_cwe_with_bert(description, embedding_model, classifier, top_cwes, threshold=0.3):
    """
    Predict CWE categories for a given CVE description using BERT embeddings
    """
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

# Main execution flow
def main(file_path, bert_model_name='all-MiniLM-L6-v2'):
    vulnerabilities, format_version = load_nvd_data(file_path)
    
    if not vulnerabilities or not format_version:
        print("No data loaded. Exiting.")
        return
    
    print(f"Processing {len(vulnerabilities)} vulnerabilities...")
    cve_cwe_pairs = extract_cve_cwe_pairs(vulnerabilities, format_version)
    print(f"Extracted {len(cve_cwe_pairs)} CVE-CWE pairs")
    
    if len(cve_cwe_pairs) == 0:
        print("No valid CVE-CWE pairs found. Check the format of your JSON file.")
        return
    
    df, top_cwes = prepare_data(cve_cwe_pairs)
    print(f"Prepared data with {len(df)} entries and {len(top_cwes)} top CWEs")
    
    if len(df) == 0:
        print("No data to train on. Exiting.")
        return
    
    embedding_model, classifier = train_model_with_bert(df, top_cwes, bert_model_name)
    
    # Save the trained model components
    print("Saving model components...")
    with open('bert_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    with open('top_cwes.pkl', 'wb') as f:
        pickle.dump(top_cwes, f)
    
    # Note: We don't pickle the embedding_model, as it's better to reload it from HuggingFace
    with open('bert_model_name.pkl', 'wb') as f:
        pickle.dump(bert_model_name, f)
    
    print("\nModel saved. Testing with a sample description...")
    
    # Test with a sample description
    sample_description = """
    A vulnerability in the web interface of Product X allows remote attackers 
    to execute arbitrary code via a crafted HTTP request. The issue occurs due 
    to improper input validation in the login form.
    """
    
    predictions = predict_cwe_with_bert(sample_description, embedding_model, classifier, top_cwes)
    
    print("\nPredicted CWEs for sample description:")
    for pred in predictions:
        print(f"{pred['cwe']}: {pred['probability']:.4f}")

# Function to load and use a saved model
def load_and_predict(description):
    import pickle
    from sentence_transformers import SentenceTransformer
    
    # Load the model components
    with open('bert_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    
    with open('top_cwes.pkl', 'rb') as f:
        top_cwes = pickle.load(f)
    
    with open('bert_model_name.pkl', 'rb') as f:
        bert_model_name = pickle.load(f)
    
    # Load the embedding model
    embedding_model = SentenceTransformer(bert_model_name)
    
    # Make a prediction
    predictions = predict_cwe_with_bert(description, embedding_model, classifier, top_cwes)
    
    return predictions

# Run the script if executed directly
if __name__ == "__main__":
    # Specify the path to your local NVD JSON file
    file_path = 'nvdcve-1.1-2024.json'
    
    # Select a BERT model - smaller models are faster but may be less accurate
    # Options:
    # - 'all-MiniLM-L6-v2' (small & fast)
    # - 'all-mpnet-base-v2' (more accurate but slower)
    # - 'paraphrase-multilingual-mpnet-base-v2' (good for multilingual data)
    # - 'msmarco-distilbert-base-v4' (tuned for information retrieval)
    bert_model_name = 'all-MiniLM-L6-v2'
    
    main(file_path, bert_model_name)
