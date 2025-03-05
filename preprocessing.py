import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re

# 1. Load the JSON data
df = pd.read_json('PreProcessCVEs.json')  # Replace 'your_data.json' with the actual file name

# 2. Data Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE) # Remove punctuation
    text = text.lower() # Lowercase
    text = [word for word in text.split() if word not in stop_words and len(word) > 2] # Remove stopwords and short words
    text = " ".join(text)
    return text

df['description'] = df['description'].apply(clean_text)

# 3. Feature Engineering (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000) # Limit features for efficiency
X = tfidf.fit_transform(df['description'])

# 4. Prepare labels
y = df['cwe_id'].astype(str)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train a Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42) # Adjust n_estimators as needed
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
print(classification_report(y_test, y_pred, zero_division=1)) # Handle potential zero division