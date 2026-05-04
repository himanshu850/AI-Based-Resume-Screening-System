from flask import Flask, request, jsonify
import pickle
from utils import clean_text

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        resume_text = request.form.get("resume") or request.get_json().get("resume")
        
        cleaned = clean_text(resume_text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        return f"Result: {'Selected' if prediction == 1 else 'Rejected'}"
    
    return "Use POST request with JSON data"
@app.route('/form')
def form():
    return '''
        <form method="post" action="/predict">
            <textarea name="resume" rows="10" cols="50"></textarea><br>
            <button type="submit">Submit</button>
        </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)