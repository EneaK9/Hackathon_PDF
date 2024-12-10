# classifier.py
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

class Classifier:
    def __init__(self):
        
        self.categories = {
            'Academic': {
                'keywords': [
                    'research', 'study', 'analysis', 'methodology', 'hypothesis',
                    'data', 'results', 'conclusion', 'abstract', 'references',
                    'journal', 'published', 'university', 'academic', 'theory',
                    'experiment', 'literature', 'scientific', 'citation',
                    'dissertation', 'thesis', 'peer-review', 'empirical',
                    'qualitative', 'quantitative', 'statistical', 'scholar',
                    'publication', 'conference', 'proceedings', 'academia',
                    'professor', 'faculty', 'research paper', 'dissertation',
                    'laboratory', 'experiment', 'findings', 'methodology',
                    'variables', 'hypothesis testing', 'literature review',
                    'doctoral', 'postdoctoral', 'symposium', 'seminar',
                    'curriculum', 'pedagogy', 'fellowship', 'grant proposal',
                    'institutional', 'scholarly', 'academic journal'
                ],
                'weight': 1.5
            },
            'Business': {
                'keywords': [
                    'company', 'market', 'business', 'profit', 'revenue',
                    'strategy', 'management', 'customer', 'sales', 'financial',
                    'corporate', 'investment', 'stockholder', 'commercial',
                    'enterprise', 'industry', 'economic', 'marketing', 'ROI',
                    'stakeholder', 'portfolio', 'assets', 'liabilities',
                    'balance sheet', 'profit margin', 'quarterly', 'fiscal',
                    'budget', 'forecast', 'merger', 'acquisition', 'startup',
                    'venture capital', 'equity', 'shareholders', 'CEO', 'CFO',
                    'operations', 'supply chain', 'logistics', 'procurement',
                    'competitive advantage', 'market share', 'B2B', 'B2C',
                    'dividend', 'valuation', 'EBITDA', 'cash flow', 'IPO',
                    'private equity', 'hedge fund', 'commodities', 'futures',
                    'derivatives', 'arbitrage', 'leverage', 'liquidity'
                ],
                'weight': 1.2
            },
            'Legal': {
                'keywords': [
                    'law', 'legal', 'contract', 'agreement', 'terms',
                    'party', 'clause', 'regulation', 'compliance', 'rights',
                    'obligation', 'court', 'jurisdiction', 'statutory', 'liability',
                    'pursuant', 'hereby', 'provision', 'legislation', 'plaintiff',
                    'defendant', 'attorney', 'counsel', 'litigation', 'arbitration',
                    'verdict', 'settlement', 'damages', 'tort', 'prosecution',
                    'judiciary', 'statute', 'precedent', 'constitutional',
                    'regulatory', 'compliance', 'mandate', 'injunction',
                    'testimony', 'negligence', 'malpractice', 'affidavit',
                    'subpoena', 'deposition', 'indemnification', 'due diligence',
                    'breach', 'intellectual property', 'patent', 'trademark'
                ],
                'weight': 1.3
            },
            'Technical': {
                'keywords': [
                    'software', 'hardware', 'system', 'database', 'network',
                    'algorithm', 'code', 'programming', 'architecture', 'interface',
                    'API', 'framework', 'development', 'implementation', 'protocol',
                    'security', 'encryption', 'bandwidth', 'server', 'client',
                    'cloud', 'infrastructure', 'deployment', 'integration',
                    'debugging', 'testing', 'version control', 'repository',
                    'backend', 'frontend', 'fullstack', 'scalability', 'latency',
                    'throughput', 'microservices', 'containerization', 'DevOps',
                    'machine learning', 'artificial intelligence', 'deep learning',
                    'neural network', 'big data', 'data mining', 'blockchain',
                    'kubernetes', 'docker', 'agile', 'scrum', 'CI/CD'
                ],
                'weight': 1.4
            },
            'Medical': {
                'keywords': [
                    'patient', 'diagnosis', 'treatment', 'clinical', 'medical',
                    'healthcare', 'hospital', 'physician', 'surgery', 'medication',
                    'prescription', 'symptoms', 'prognosis', 'pathology', 'therapy',
                    'pharmaceutical', 'chronic', 'acute', 'outpatient', 'inpatient',
                    'anatomy', 'physiology', 'oncology', 'cardiology', 'neurology',
                    'pediatrics', 'psychiatric', 'surgical', 'diagnostic',
                    'therapeutic', 'immunology', 'epidemiology', 'vaccination',
                    'rehabilitation', 'emergency', 'trauma', 'intensive care',
                    'radiology', 'pharmacology', 'anesthesia', 'endocrinology',
                    'orthopedic', 'geriatric', 'palliative', 'preventive'
                ],
                'weight': 1.4
            },
            'Engineering': {
                'keywords': [
                    'design', 'construction', 'manufacturing', 'specifications',
                    'mechanical', 'electrical', 'civil', 'structural', 'industrial',
                    'robotics', 'automation', 'CAD', 'simulation', 'prototype',
                    'materials', 'assembly', 'quality control', 'maintenance',
                    'safety standards', 'ISO', 'compliance', 'efficiency',
                    'optimization', 'thermal', 'fluid dynamics', 'stress analysis',
                    'fabrication', 'tooling', 'machining', 'welding', 'inspection',
                    'tolerances', 'engineering drawings', 'schematics', 'blueprints',
                    'aerodynamics', 'biomechanical', 'chemical engineering',
                    'control systems', 'HVAC', 'metallurgy', 'tribology'
                ],
                'weight': 1.3
            },
            'Finance': {
                'keywords': [
                    'banking', 'investment', 'trading', 'securities', 'bonds',
                    'stocks', 'mutual funds', 'options', 'forex', 'cryptocurrency',
                    'portfolio management', 'risk assessment', 'hedge', 'dividend',
                    'interest rate', 'capital gains', 'market analysis', 'broker',
                    'derivative', 'futures contract', 'asset allocation', 'bear market',
                    'bull market', 'credit rating', 'diversification', 'equity',
                    'fixed income', 'inflation', 'liquidity', 'margin', 'volatility',
                    'yield', 'beta', 'alpha', 'price-to-earnings', 'book value'
                ],
                'weight': 1.3
            },
            'Environmental': {
                'keywords': [
                    'sustainability', 'renewable energy', 'carbon footprint',
                    'climate change', 'biodiversity', 'ecosystem', 'conservation',
                    'environmental impact', 'green technology', 'recycling',
                    'waste management', 'pollution', 'emissions', 'solar power',
                    'wind energy', 'geothermal', 'hydroelectric', 'biomass',
                    'ecological', 'environmental protection', 'habitat restoration',
                    'greenhouse gas', 'carbon neutral', 'sustainable development',
                    'environmental compliance', 'wildlife conservation'
                ],
                'weight': 1.2
            },
            'Government': {
                'keywords': [
                    'policy', 'regulation', 'legislation', 'government agency',
                    'federal', 'state', 'municipal', 'public sector', 'bureaucracy',
                    'administrative', 'executive order', 'public policy', 'governance',
                    'referendum', 'election', 'congressional', 'parliamentary',
                    'diplomatic', 'foreign policy', 'domestic policy', 'legislature',
                    'judiciary', 'constitutional', 'democracy', 'republic',
                    'civil service', 'public administration', 'political'
                ],
                'weight': 1.2
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
        max_score = max(category_scores.values())
        second_max = sorted(category_scores.values(), reverse=True)[1] if len(category_scores) > 1 else 0
        
        # Calculate confidence based on difference between top scores
        confidence = (max_score - second_max) / max_score * 100
        
        # Adjust confidence to be between 0 and 100
        confidence = min(max(confidence, 0), 100)
        
        return round(confidence, 2)

    def classify_document(self, text):
        
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
