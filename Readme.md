# CVE to CWE Prediction Tool

This tool uses machine learning to predict Common Weakness Enumeration (CWE) categories based on Common Vulnerabilities and Exposures (CVE) descriptions.

## Features

- Processes CVE data from NVD JSON files
- Extracts and analyzes CVE descriptions and their associated CWE IDs
- Trains a multi-label classification model using TF-IDF and Logistic Regression
- Provides a web interface for easy prediction of CWEs from vulnerability descriptions
- Links to official CWE documentation for predicted categories

## Requirements

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - flask

## Setup and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Before using the web application, you need to train the model:

```bash
python train_model.py
```

This script will:
- Load NVD data from the local file 'nvdcve-1.1-2024.json'
- Extract CVE-CWE pairs
- Train and evaluate the classification model
- Save the model components to disk

### 3. Run the Web Application

```bash
python app.py
```

Then open your web browser and go to http://127.0.0.1:5000/

### 4. Use the API

You can also use the prediction API directly:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"description":"A SQL injection vulnerability in login.php allows attackers to bypass authentication."}'
```

## Improving the Model

To improve the model's performance:

1. **Add More Data**: Use multiple years of NVD data
2. **Try Advanced Embeddings**: Experiment with Word2Vec or BERT embeddings
3. **Balance Classes**: Implement techniques like SMOTE for rare CWEs
4. **Advanced Models**: Try neural networks or ensemble methods

## Files

- `train_model.py`: Script to load data, train and save the model
- `app.py`: Flask web application
- `templates/`: HTML templates for the web interface
- `requirements.txt`: Required Python packages
- Model files (generated after training):
  - `cve_cwe_vectorizer.pkl`: The TF-IDF vectorizer
  - `cve_cwe_classifier.pkl`: The trained classifier
  - `top_cwes.pkl`: List of the top CWE categories


# Generated with Claude
This application was primarily developed by Claude's 3.7 Sonnet AI model
