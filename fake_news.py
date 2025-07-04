import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Datasets
fake_df = pd.read_csv('dataset/Fake.csv')
true_df = pd.read_csv('dataset/True.csv')

# Add Labels
fake_df['label'] = 'FAKE'
true_df['label'] = 'REAL'

# Combine Data
df = pd.concat([fake_df, true_df], axis=0)
print(f"Combined Dataset Shape: {df.shape}")

# Shuffle the Data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check Columns
print(df.columns)

# Features and Labels
X = df['text']     # News content
y = df['label']    # FAKE or REAL

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Prediction
y_pred = model.predict(tfidf_test)

# Evaluation
score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {round(score*100, 2)}%")
print("\nConfusion Matrix:")
print(cm)

# Optional: Test with custom news text
while True:
    user_input = input("\nEnter news text to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    test_vector = vectorizer.transform([user_input])
    prediction = model.predict(test_vector)
    print(f"Prediction: {prediction[0]}")
