import random
import string
import joblib
from flask import Flask, request, jsonify
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
response_map = joblib.load("response.pkl")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = text.lower()
    text = text.replace('-', ' ')
    text = "".join(char for char in text if char not in string.punctuation)
    stemmed_text = stemmer.stem(text)
    tokens = stemmed_text.split()
    return " ".join(tokens)

def get_response(message, threshold=0.4):
    clean_input = preprocess(message)
    vectorized_input = vectorizer.transform([clean_input])
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_input)[0]
        max_prob = max(probabilities)
        predicted_index = probabilities.argmax()
        predicted_tag = model.classes_[predicted_index]

        print(f"[DEBUG] Input: {message}")
        print(f"[DEBUG] Cleaned: {clean_input}")
        print(f"[DEBUG] Probabilities: {probabilities}")
        print(f"[DEBUG] Max Prob: {max_prob}, Predicted Tag: {predicted_tag}")

        if max_prob > threshold:
            return random.choice(response_map[predicted_tag])
        else:
            return "Maaf, saya belum mempelajari hal tersebut."
    else:
        predicted_tag = model.predict(vectorized_input)[0]
        return random.choice(response_map[predicted_tag])

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message")
    if not message:
        return jsonify({"code": 400, "message": "Message cannot be empty!"})
    
    reply = get_response(message)
    return jsonify({
        "code": 200,
        "message": "Success",
        "data": {
            "response": reply
        }
    })

if __name__ == "__main__":
    app.run(debug=True)