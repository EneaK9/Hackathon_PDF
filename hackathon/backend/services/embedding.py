from sklearn.feature_extraction.text import TfidfVectorizer

class Embedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def transform(self, text):
        return self.vectorizer.fit_transform([text]).toarray()
