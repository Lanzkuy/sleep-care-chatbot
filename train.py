import json
import string
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def load_data(json_path):
    with open(json_path) as json_data:
        data = json.load(json_data)
    labels, texts, response = [], [], {}
    for intent in data:
        for pattern in intent['pattern']:
            labels.append(intent['tag'])
            texts.append(pattern)
        response[intent['tag']] = intent['response']
    return labels, texts, response

def preprocess(text):
    text = text.lower()
    text = text.replace('-', ' ')
    text = "".join(char for char in text if char not in string.punctuation)
    stemmed_text = stemmer.stem(text)
    tokens = stemmed_text.split()
    return " ".join(tokens)
    
def train_and_save_model():
    path = "./intents.json"
    labels, texts, response = load_data(path)
    texts = [preprocess(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    vectorizer = CountVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
    X_train_vector = vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_vector, y_train)
    y_pred = model.predict(X_test_vector)

    print("===== Evaluation =====")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(response, "response.pkl")
    
    print("Model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()