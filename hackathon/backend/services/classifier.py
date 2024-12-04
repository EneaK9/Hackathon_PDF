# classifier.py
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

class Classifier:
    def __init__(self):
        # Define category keywords and their weights
        self.categories = {
            'Academic': {
                'keywords': [
                    'research', 'study', 'analysis', 'methodology', 'hypothesis',
                    'data', 'results', 'conclusion', 'abstract', 'references',
                    'journal', 'published', 'university', 'academic', 'theory',
                    'experiment', 'literature', 'scientific', 'citation'
                ],
                'weight': 1.5
            },
            'Business': {
                'keywords': [
                    'company', 'market', 'business', 'profit', 'revenue',
                    'strategy', 'management', 'customer', 'sales', 'financial',
                    'corporate', 'investment', 'stockholder', 'commercial', 'trade',
                    'enterprise', 'industry', 'economic'
                ],
                'weight': 1.2
            },
            'Legal': {
                'keywords': [
                    'law', 'legal', 'contract', 'agreement', 'terms',
                    'party', 'clause', 'regulation', 'compliance', 'rights',
                    'obligation', 'court', 'jurisdiction', 'statutory', 'liability',
                    'pursuant', 'hereby', 'provision'
                ],
                'weight': 1.3
            }
        }
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000
        )

    def preprocess_text(self, text):
        """Clean and preprocess the text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def calculate_confidence(self, category_scores):
        """Calculate confidence score based on category scores"""
        max_score = max(category_scores.values())
        second_max = sorted(category_scores.values(), reverse=True)[1] if len(category_scores) > 1 else 0
        
        # Calculate confidence based on difference between top scores
        confidence = (max_score - second_max) / max_score * 100
        
        # Adjust confidence to be between 0 and 100
        confidence = min(max(confidence, 0), 100)
        
        return round(confidence, 2)

    def classify_document(self, text):
        """
        Classify the document and return category with confidence score
        """
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return "Unknown", 0.0

        # Calculate scores for each category
        category_scores = {}
        
        # Convert text to TF-IDF vector
        text_vector = self.vectorizer.fit_transform([processed_text])
        terms = self.vectorizer.get_feature_names_out()
        
        # Calculate score for each category
        for category, info in self.categories.items():
            score = 0
            weight = info['weight']
            
            # Count keyword matches with TF-IDF weights
            for keyword in info['keywords']:
                if keyword in terms:
                    idx = list(terms).index(keyword)
                    score += text_vector.toarray()[0][idx] * weight
            
            # Normalize score by number of keywords
            category_scores[category] = score / len(info['keywords'])

        # Get category with highest score
        if not category_scores:
            return "Unknown", 0.0
            
        best_category = max(category_scores.items(), key=lambda x: x[1])[0]
        confidence = self.calculate_confidence(category_scores)

        return best_category, confidence

    def get_detailed_analysis(self, text):
        """
        Get detailed analysis of why the classification was made
        """
        processed_text = self.preprocess_text(text)
        matches = {}
        
        for category, info in self.categories.items():
            category_matches = []
            for keyword in info['keywords']:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', processed_text))
                if count > 0:
                    category_matches.append((keyword, count))
            matches[category] = category_matches
            
        return matches
