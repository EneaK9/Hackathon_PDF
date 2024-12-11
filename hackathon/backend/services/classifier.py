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
                    'author', 'journal', 'published', 'university', 'academic', 
                    'theory', 'experiment', 'literature', 'scientific', 'citation', 
                    'peer', 'review', 'dissertation', 'thesis', 'peer-review', 'empirical', 'survey', 
                    'qualitative', 'quantitative', 'statistical', 'scholar', 'professor',
                    'publication', 'conference', 'proceedings', 'academia', 'postgraduate',
                    'professor', 'faculty', 'research paper', 'laboratory', 'findings', 
                    'researcher', 'variables', 'hypothesis testing', 'literature review', 'citations',
                    'doctoral', 'postdoctoral', 'symposium', 'seminar', 'lecture', 
                    'curriculum', 'pedagogy', 'fellowship', 'grant proposal', 'grant',
                    'institutional', 'scholarly', 'academic journal', 'peer-reviewed', 'manuscript'
                    'archive', 'repository', 'citation index', 'bibliometrics', 'bibliography',
                    'bibliography', 'case study', 'academic integrity', 'scholarly work', 'academic network', 
                    'academic publishing', 'open access', 'academic discourse', 'research framework', 
                    'interdisciplinary', 'fieldwork', 'academic rigor', 'theoretical framework', 'scholarly communication', 
                    'educational policy', 'intellectual property', 'ethical standards', 'academic assessment', 'research ethics', 
                    'qualitative analysis', 'quantitative analysis', 'academic collaboration', 'meta-analysis', 'research design', 
                    'academic symposium', 'research innovation', 'academic contribution', 'experimental design', 'academic mentor', 
                    'academic journal club', 'research methodology', 'empirical research', 'academic analytics', 'research outputs', 
                    'academic indexing', 'research funding', 'academic conference', 'research impact', 'academic reputation', 
                    'research dissemination', 'knowledge dissemination', 'research validation', 'academic citation', 'research collaboration', 
                    'academic recognition', 'plagiarism', 'peer interaction', 'open science', 'research lifecycle'
                ],
                'weight': 1.5
            },
            'Business': {
                'keywords': [
                    'company', 'market', 'business', 'profit', 'revenue', 
                    'strategy', 'management', 'customer', 'sales', 'financial', 
                    'corporate', 'investment', 'stockholder', 'commercial', 'growth',
                    'enterprise', 'industry', 'economic', 'marketing', 'ROI',
                    'stakeholder', 'portfolio', 'assets', 'liabilities','balance sheet',
                    'profit margin', 'quarterly', 'fiscal', 'budget', 'forecast', 
                    'merger', 'acquisition', 'startup', 'venture capital', 'equity', 
                    'shareholders', 'CEO', 'CFO', 'operations', 'supply chain', 
                    'logistics', 'procurement', 'competitive advantage', 'market share', 'B2B', 
                    'B2C', 'dividend', 'valuation', 'EBITDA', 'cash flow', 
                    'IPO', 'private equity', 'hedge fund', 'commodities', 'futures',
                    'derivatives', 'arbitrage', 'leverage', 'liquidity', 'solvent',
                    'insolvent', 'bankruptcy', 'restructuring', 'turnaround', 'reorganization',
                    'corporate governance', 'board of directors', 'shareholder value', 'business model',
                    'strategic planning', 'business development', 'market research', 'competitive analysis',
                    'SWOT analysis', 'business process', 'business intelligence', 'financial analysis',
                    'financial statement', 'income statement', 'cash flow statement', 'balance sheet',
                    'financial planning', 'financial forecasting', 'financial management', 'financial risk',
                    'financial performance', 'financial reporting', 'financial strategy', 'financial modeling',
                    'financial audit', 'financial compliance', 'financial regulation', 'financial institution',
                    'financial market', 'financial services', 'financial technology', 'financial crisis', 'financial stability',
                    'stability', 'financial innovation', 'financial inclusion', 'financial literacy', 'financial education',
                    'financial advisor', 'financial consultant', 'financial planner', 'financial analyst', 'financial economist'
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
                    'judiciary', 'statute', 'precedent', 'constitutional', 'regulatory', 
                    'compliance', 'mandate', 'injunction', 'testimony', 'negligence', 
                    'portfolio', 'study', 'fiduciary duty', 'judicial process', 'dividend',
                    'review', 'financial', 'hypothesis', 'sales', 'hypothesis testing',
                    'case law', 'amicus brief', 'real estate law', 'arbitrage', 'academic integrity',
                    'research design', 'academic journal club', 'research translation', 'citations', 'international law',
                    'research', 'growth', 'legal precedent', 'industry', 'advocacy', 
                    'legal obligation', 'criminal law', 'academic network', 'environmental law', 'fellowship',
                    'legal consultation', 'academic journal', 'liquidity', 'trade law', 'acquisition', 
                    'archive', 'peer-reviewed', 'academic', 'appeal process', 'institutional', 
                    'pedagogy', 'legal framework', 'equity', 'contract negotiation', 'contract drafting', 
                    'regulatory affairs', 'contractual terms', 'postdoctoral', 'equity law', 'copyright', 
                    'civil law', 'empirical research', 'intellectual rights', 'theoretical framework', 'conclusion', 
                    'procurement', 'family law', 'court hearing', 'customer', 'management', 
                    'legal liability', 'corporate law', 'literature', 'profit', 'legal compliance',
                    'labor law', 'legal research', 'legal advice', 'legal counsel', 'legal representation',
                    'legal document', 'legal dispute', 'legal system', 'legal code', 'legal procedure'
                ],
                'weight': 1.3
            },
            'Technical': {
                'keywords': [
                    'software', 'hardware', 'system', 'database', 'network',
                    'algorithm', 'code', 'programming', 'architecture', 'interface',
                    'API', 'framework', 'development', 'implementation', 'protocol',
                    'security', 'encryption', 'bandwidth', 'server', 'client',
                    'cloud', 'infrastructure', 'deployment', 'integration', 'debugging', 
                    'testing', 'version control', 'repository', 'backend', 'frontend', 
                    'fullstack', 'scalability', 'latency', 'throughput', 'microservices', 
                    'containerization', 'DevOps', 'machine learning', 'artificial intelligence', 'deep learning',
                    'neural network', 'big data', 'data mining', 'blockchain', 'kubernetes', 
                    'docker', 'agile', 'scrum', 'CI/CD', 'virtualization', 
                    'cybersecurity', 'data analytics', 'automation', 'IoT', 'edge computing', 
                    'quantum computing', 'API Gateway', 'load balancing', 'middleware', 'runtime', 
                    'system optimization', 'parallel processing', 'high availability', 'fault tolerance', 'performance tuning', 
                    'data visualization', 'distributed systems', 'storage systems', 'serverless computing', 'SRE',
                    'observability', 'logging', 'monitoring', 'incident management', 'workflow automation', 
                    'container orchestration', 'predictive analytics', 'natural language processing', 'computer vision', 'reinforcement learning',
                    'knowledge graph', 'ETL', 'API integration', 'CI tools', 'configuration management', 
                    'source control', 'infrastructure as code', 'testing frameworks', 'software lifecycle', 'technical debt',
                    'software design patterns', 'clean code', 'code review', 'semantic versioning', 'RESTful services', 
                    'GraphQL', 'WebSocket', 'authentication', 'authorization', 'OAuth', 
                    'OpenID', 'SAML', 'PKI', 'cryptography', 'firewall', 
                    'intrusion detection', 'security audit', 'security policy', 'security incident', 'security breach', 
                    'security patch', 'security compliance', 'security awareness', 'security best practices', 'security architecture'
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
                    'pediatrics', 'psychiatric', 'surgical', 'diagnostic', 'therapeutic', 
                    'immunology', 'epidemiology', 'vaccination', 'rehabilitation', 'emergency', 
                    'trauma', 'intensive care', 'radiology', 'pharmacology', 'anesthesia', 
                    'endocrinology', 'orthopedic', 'geriatric', 'palliative', 'preventive',
                    'telemedicine', 'biopsy', 'prognostic', 'hematology', 'dermatology',
                    'neonatology', 'toxicology', 'urology', 'otolaryngology', 'rheumatology',
                    'gastroenterology', 'infectious diseases', 'genomics', 'clinical trials', 'primary care', 
                    'maternal health', 'preventive care', 'nutritional health', 'reproductive health', 'health informatics', 
                    'pain management', 'critical care', 'pathogenesis', 'medical ethics', 'biosafety', 
                    'public health', 'global health', 'health disparities', 'evidence-based medicine', 'alternative medicine',
                    'complementary therapies', 'biotechnology', 'nanomedicine', 'regenerative medicine', 'clinical guidelines', 
                    'health promotion', 'mental health', 'occupational health', 'sports medicine', 'audiology', 
                    'medical imaging', 'bioinformatics', 'pharmacovigilance', 'medical devices', 'medical technology',
                    'medical research', 'medical education', 'medical training', 'medical practice', 'medical profession',
                    'medical association', 'medical journal', 'medical conference', 'medical literature', 'medical terminology',
                    'medical transcription', 'medical coding', 'medical billing', 'medical records', 'medical history',
                    'medical examination', 'medical consultation', 'medical diagnosis', 'medical treatment', 'medical procedure',
                    'genetics', 'microbiology', 'telehealth', 'pathophysiology', 'biostatistics'
                ],
                'weight': 1.4
            },
            'Engineering': {
                'keywords': [
                    'design', 'construction', 'manufacturing', 'specifications', 'mechanical', 
                    'electrical', 'civil', 'structural', 'industrial', 'robotics', 
                    'automation', 'CAD', 'simulation', 'prototype', 'materials', 
                    'assembly', 'quality control', 'maintenance', 'safety standards', 'ISO', 
                    'compliance', 'efficiency', 'optimization', 'thermal', 'fluid dynamics', 
                    'stress analysis', 'fabrication', 'tooling', 'machining', 'welding', 
                    'inspection', 'tolerances', 'engineering drawings', 'schematics', 'blueprints',
                    'aerodynamics', 'biomechanical', 'chemical engineering', 'control systems', 'HVAC', 
                    'metallurgy', 'tribology', '3D printing', 'acoustics', 'additive manufacturing', 
                    'advanced manufacturing', 'aerospace engineering', 'aerospace materials', 'agricultural engineering', 'air quality management', 
                    'alternative fuels', 'aqua engineering', 'artificial intelligence', 'automation systems', 'autonomous systems',
                    'biodegradable materials', 'biodynamics', 'bioelectronics', 'bioengineering', 'bioinformatics', 
                    'bioinspired engineering', 'biomimetics', 'biotechnology', 'bridge design', 'catalysis', 
                    'chemical kinetics', 'circuit design', 'climate modeling', 'composite materials', 'computational modeling',
                    'computer-aided engineering', 'construction management', 'control algorithms', 'cyber-physical systems', 'cybersecurity in engineering', 
                    'data analysis', 'digital twins', 'distributed control systems', 'distributed energy resources', 'dynamics', 
                    'earthquake engineering', 'electric power distribution', 'electric propulsion', 'electric vehicles', 'electrochemical systems',
                    'electromagnetics', 'embedded systems', 'energy efficiency', 'energy harvesting', 'energy storage', 
                    'energy systems', 'environmental engineering', 'environmental impact analysis', 'environmental remediation', 'ergonomics',
                    'failure analysis', 'finite element analysis', 'fluid mechanics', 'forensic engineering', 'fusion energy', 
                    'geotechnical', 'geothermal systems', 'graphene applications', 'green building materials', 'heat transfer',
                    'high-performance computing', 'human factors engineering', 'hydraulics', 'industrial automation', 'industrial internet of things (IIoT)',
                    'thermal imaging', 'thermal management', 'thermodynamics', 'traffic engineering', 'transportation engineering'
                ],
                'weight': 1.3
            },
            'Finance': {
                'keywords': [
                    'banking', 'investment', 'trading', 'securities', 'bonds',
                    'stocks', 'mutual funds', 'options', 'forex', 'cryptocurrency',
                    'portfolio management', 'risk assessment', 'hedge', 'dividend', 'interest rate', 
                    'capital gains', 'market analysis', 'broker', 'derivative', 'futures contract', 
                    'asset allocation', 'bear market', 'bull market', 'credit rating', 'diversification', 
                    'equity', 'fixed income', 'inflation', 'liquidity', 'margin', 
                    'volatility', 'yield', 'beta', 'alpha', 'price-to-earnings', 
                    'book value', 'wealth management', 'financial planning', 'private equity', 'venture capital', 
                    'leveraged buyout', 'cash flow', 'monetary policy', 'debt management', 'financial derivatives', 
                    'real estate investment', 'corporate finance', 'economic forecasting', 'cost of capital', 'financial modeling', 
                    'valuation', 'credit risk', 'financial reporting', 'treasury management', 'payment systems', 
                    'financial inclusion', 'sovereign debt', 'structured finance', 'mergers and acquisitions', 'profitability', 
                    'fiscal policy', 'taxation', 'budgeting', 'hedge funds', 'index funds', 
                    'retirement planning', 'actuarial science', 'financial engineering', 'economic indicators', 'securitization', 
                    'microfinance', 'cash reserve ratio', 'alternative investments', 'arbitrage', 'balance sheet', 
                    'blockchain', 'capital budgeting', 'cash management', 'collateral', 'commodities', 
                    'credit default swap', 'currency exchange', 'debt consolidation', 'discount rate', 'earnings per share', 
                    'exchange-traded funds', 'financial leverage', 'forensic accounting', 'global markets', 'initial public offering', 
                    'intangible assets', 'interest income', 'investment banking', 'liability management', 'market liquidity', 
                    'monetary assets', 'operating income', 'over-the-counter', 'personal finance', 'pricing strategy', 
                    'principal repayment', 'profit margin', 'proprietary trading', 'quantitative analysis', 'real return', 
                    'revenue streams', 'shareholder equity', 'socially responsible investing', 'technical analysis', 'term structure', 
                    'treasury bonds', 'underwriting', 'global finance', 'goodwill', 'insurance premiums', 
                    'internal rate of return', 'investment grade', 'liquidation value', 'monetary equilibrium', 'non-performing assets', 
                    'offshore banking', 'peer-to-peer lending', 'price elasticity', 'profit and loss', 'return on equity'
                ],
                'weight': 1.3
            },
            'Environmental': {
                'keywords': [
                    'sustainability', 'renewable energy', 'carbon footprint', 'climate change', 'biodiversity', 
                    'ecosystem', 'conservation', 'environmental impact', 'green technology', 'recycling',
                    'waste management', 'pollution', 'emissions', 'solar power', 'wind energy', 
                    'geothermal', 'hydroelectric', 'biomass', 'ecological', 'environmental protection', 
                    'habitat restoration', 'greenhouse gas', 'carbon neutral', 'sustainable development', 'environmental compliance', 
                    'wildlife conservation', 'urban forestry', 'water conservation', 'air quality', 'organic farming',
                    'energy efficiency', 'climate resilience', 'eco-friendly practices', 'deforestation', 'reforestation', 
                    'ocean conservation', 'natural resources', 'environmental awareness', 'plastic reduction', 'clean energy', 
                    'sustainable agriculture', 'carbon offset', 'green building', 'environmental justice', 'microplastics',
                    'circular economy', 'zero waste', 'nature preservation', 'soil health', 'renewable resources', 
                    'environmental restoration', 'marine biodiversity', 'sustainable fishing', 'alternative energy', 'energy transition', 
                    'eco-tourism', 'carbon sequestration', 'fossil fuel alternatives', 'water purification', 'forest management',
                    'ecosystem services', 'environmental ethics', 'environmental policy', 'habitat conservation', 'low-impact living', 
                    'renewable infrastructure', 'environmental innovation', 'waste-to-energy', 'permaculture', 'sustainable forestry', 
                    'environmental education', 'carbon budgeting', 'land restoration', 'resource efficiency', 'natural habitats',
                    'environmental sustainability', 'energy conservation', 'water efficiency', 'environmental stewardship', 'wetland preservation', 
                    'green infrastructure', 'environmental advocacy', 'organic waste', 'native species protection', 'urban greening', 
                    'energy storage', 'carbon trading', 'low-carbon economy', 'smart grids', 'bioenergy', 
                    'aquatic ecosystems', 'air purification', 'environmental monitoring', 'eco-certification', 'climate adaptation',
                    'carbon capture', 'habitat corridors', 'environmental remediation', 'sustainable cities', 'renewable innovations', 
                    'electric vehicles', 'alternative fuels', 'environmental equity', 'green logistics', 'cultural landscapes', 
                    'eco-innovation', 'environmental health', 'natural capital', 'blue carbon', 'energy justice', 
                    'sustainable materials', 'water recycling', 'soil carbon', 'eco-conscious design', 'green manufacturing',
                    'nature-based solutions', 'wildlife corridors', 'low-emission zones', 'clean transportation', 'plant-based solutions'
                ],
                'weight': 1.2
            },
            'Government': {
                'keywords': [
                    'policy', 'regulation', 'legislation', 'government agency', 'federal', 
                    'state', 'municipal', 'public sector', 'bureaucracy', 'administrative', 
                    'executive order', 'public policy', 'governance', 'referendum', 'election', 
                    'congressional', 'parliamentary', 'diplomatic', 'foreign policy', 'domestic policy', 
                    'legislature', 'judiciary', 'constitutional', 'democracy', 'republic',
                    'civil service', 'public administration', 'political', 'cabinet', 'executive branch', 
                    'judicial system', 'political science', 'civic engagement', 'statute', 'ordinance', 
                    'fiscal policy', 'taxation', 'administrative law', 'public welfare', 'national security',
                    'transparency', 'accountability', 'sovereignty', 'international relations', 'federalism', 
                    'statecraft', 'policy analysis', 'public institutions', 'rule of law', 'citizenship', 
                    'e-governance', 'non-governmental', 'public affairs', 'diplomacy', 'voter registration', 
                    'campaign finance', 'administration', 'sovereign state', 'judicial review', 'executive leadership',
                    'political institutions', 'lawmaking', 'civic duty', 'ministerial', 'governing body', 
                    'federal agencies', 'policy-making', 'political authority', 'constitutional law', 'intergovernmental', 
                    'public service', 'political ideology', 'social contract', 'centralized government', 'decentralization', 
                    'national interest', 'policy implementation', 'legislative process', 'regulatory body', 'public accountability',
                    'social governance', 'administrative procedures', 'public order', 'civic rights', 'state sovereignty', 
                    'government oversight', 'interagency cooperation', 'political participation', 'national budget', 'public sector reforms', 
                    'civic leadership', 'governmental powers', 'public office', 'policy evaluation', 'local governance', 
                    'citizen engagement', 'constitutional amendments', 'political frameworks', 'institutional governance', 'state administration',
                    'executive council', 'policy advocacy', 'public discourse', 'government mandates', 'public representation',
                    'bureaucratic oversight', 'political governance', 'statutory authority', 'government initiatives', 'policy frameworks', 
                    'statecraft strategy', 'federal oversight', 'public accountability mechanisms', 'national administration', 'government transparency'
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
