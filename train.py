import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from utils import clean_text

# Sample dataset
data = {
    "resume": [
        "Python developer with machine learning experience",
        "Nurse with patient care experience",
        "Data analyst with SQL and Excel"
    ],
    "label": [1, 0, 1]  # 1 = Relevant, 0 = Not Relevant
}

df = pd.DataFrame(data)
df['resume'] = df['resume'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['resume'])

model = LogisticRegression()
model.fit(X, df['label'])

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))