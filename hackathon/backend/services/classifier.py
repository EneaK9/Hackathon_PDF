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
                    'qualitative analysis', 'quantitative analysis', 'academic collaboration', 'meta-analysis', 'research design'
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
                    'financial planning', 'financial forecasting', 'financial management', 'financial risk'
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
                    'regulatory affairs', 'contractual terms', 'postdoctoral', 'equity law', 'copyright'
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
                    'source control', 'infrastructure as code', 'testing frameworks', 'software lifecycle', 'technical debt'
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
                    'medical imaging', 'bioinformatics', 'pharmacovigilance', 'medical devices', 'medical technology'
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
                    'electromagnetics', 'embedded systems', 'energy efficiency', 'energy harvesting', 'energy storage'
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
                    'monetary assets', 'operating income', 'over-the-counter', 'personal finance', 'pricing strategy'
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
                    'aquatic ecosystems', 'air purification', 'environmental monitoring', 'eco-certification', 'climate adaptation'
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
                ],
                'weight': 1.2
            },
            'Education' : {
                'keywords': [  
                    'teaching', 'learning', 'classroom', 'student', 'teacher',  
                    'curriculum', 'pedagogy', 'syllabus', 'assessment', 'academic achievement',  
                    'lesson planning', 'school', 'instructional methods', 'early childhood', 'student engagement',  
                    'educational reform', 'teacher training', 'homework policies', 'educational leadership', 'student success',  
                    'cognitive development', 'teaching models', 'critical pedagogy', 'educational technology tools', 'student assessment methods',  
                    'career readiness', 'literacy programs', 'numeracy', 'hands-on learning', 'group collaboration',  
                    'peer tutoring', 'extracurriculars', 'academic advisement', 'distance learning programs', 'e-learning strategies',  
                    'formative learning', 'special education needs', 'inclusive learning environments', 'blended learning strategies', 'teacher-student interaction',  
                    'personalized learning', 'flipped classroom', 'project-based instruction', 'vocational training initiatives', 'STEM initiatives',  
                    'educational workshops', 'digital literacy', 'teaching certificates', 'student well-being', 'school governance',  
                    'learning behavior', 'student portfolios', 'teacher evaluations', 'educational budgets', 'school infrastructure',  
                    'educational metrics', 'alternative schooling', 'gifted programs', 'educational equity', 'policy frameworks',  
                    'dual-enrollment programs', 'student scholarships', 'teacher pay', 'lesson evaluation', 'early interventions',  
                    'universal pre-k', 'cultural responsiveness', 'co-teaching', 'standardized curriculum', 'teacher mentoring',  
                    'education summits', 'student competitions', 'school choice', 'academic coaching', 'non-traditional learners',  
                    'school district policies', 'educational games', 'professional learning communities', 'literature-based curriculum', 'academic peer networks'  
                ],  
                'weight': 1.3  
            },
            'Technology' :{
                'keywords': [  
                    'emerging technologies', 'virtual reality', 'augmented reality', 'mobile development', 'AI integration',  
                    'machine vision', 'data warehousing', 'autonomous drones', 'smart cities', 'wearable devices',  
                    'digital twins', 'serverless platforms', 'IT management', 'code optimization', 'database management',  
                    'cloud architecture', 'data pipelines', 'computing frameworks', 'cybersecurity threats', 'blockchain applications',  
                    'IoT ecosystems', 'quantum encryption', 'neural processors', 'bioinformatics tools', 'device innovation',  
                    'programmatic advertising', 'natural user interfaces', 'speech-to-text technology', 'visual processing', 'high-performance computing',  
                    'container security', 'AI fairness', 'program synthesis', 'augmented decision-making', 'automated deployment',  
                    'digital ecosystems', '5G applications', 'wearable diagnostics', 'haptic feedback', 'user data governance',  
                    'end-to-end encryption', 'data provenance', 'dynamic content delivery', 'edge AI', 'distributed learning',  
                    'data locality', 'deep neural networks', 'hybrid clouds', 'interoperability standards', 'secure boot systems',  
                    'realtime analytics', 'responsive APIs', 'scalable algorithms', 'cross-platform apps', 'sensor fusion',  
                    'context-aware systems', 'adaptive interfaces', 'technology foresight', 'human-computer synergy', 'real-time communication',  
                    'semantic web technologies', 'smart algorithms', 'ethics in AI', 'functional programming', 'data integration strategies',  
                    'software-defined storage', 'energy-efficient computing', 'digital currencies', 'smart mobility', 'privacy-enhancing technology',  
                    'information access control', 'system virtualization', 'open platform initiatives', 'human augmentation', 'ubiquitous computing'  
                ],  
                'weight': 1.3  
            },
            'Art and Design':{
                'keywords': [  
                    'illustrative storytelling', 'conceptual sketches', 'mixed media art', 'artistic innovation', 'immersive installations',  
                    'hand-drawn techniques', 'digital palettes', 'motion studies', 'aesthetic theory', 'spatial design',  
                    'light installations', 'pattern crafting', 'iconography', 'minimalist design', 'narrative-driven art',  
                    'interactive storytelling', 'collaborative art projects', 'portfolio curation', 'art for social change', 'site-specific art',  
                    'kinetic sculptures', 'user-centered aesthetics', 'eco-friendly designs', 'materials exploration', 'vivid color contrasts',  
                    'monochromatic themes', 'historic art movements', 'sustainable design solutions', 'art direction methods', 'visual consistency',  
                    'brand storytelling', 'photo-editing tools', 'design sprints', 'style guides', 'identity branding',  
                    'experimental typography', 'real-world prototyping', 'art criticism essays', 'studio-based practices', 'freestyle painting',  
                    'dynamic forms', 'live art performances', 'sculptural installations', 'pixel-perfect editing', 'design guidelines',  
                    'audiovisual exploration', 'collaborative exhibits', 'material transparency', 'responsive illustrations', 'artistic impact',  
                    'visionary compositions', 'design ecosystems', 'future-focused design', 'human-centered approaches', 'design resilience',  
                    'light and shadow dynamics', 'material reuse', 'non-traditional formats', 'immersive theater', 'printmaking techniques',  
                    'post-modern aesthetics', 'cross-disciplinary practices', 'symbolic representations', 'data-driven designs', 'impressionist influences',  
                    'artistic agility', 'heritage-inspired designs', 'contemporary adaptations', 'media art criticism', 'emotion-driven art',  
                    'functional aesthetics', 'aesthetic minimalism', 'global art perspectives', 'universal design concepts', 'high-impact art'  
                ],  
                'weight': 1.2  
            },
            'Sport' :{
                'keywords': [  
                    'athletics', 'competition', 'training', 'coaching', 'teamwork',  
                    'strategy', 'gameplay', 'fitness', 'endurance', 'agility',  
                    'speed', 'strength', 'mental toughness', 'sportsmanship', 'exercise',  
                    'tournament', 'championship', 'league', 'match', 'recreation',  
                    'outdoor activities', 'indoor sports', 'team sports', 'individual sports', 'sports gear',  
                    'fitness tracking', 'workout routines', 'skill development', 'game analysis', 'performance metrics',  
                    'sports nutrition', 'rehabilitation', 'injury prevention', 'physical therapy', 'aerobic exercises',  
                    'anaerobic exercises', 'cross-training', 'coaching philosophy', 'playbook', 'sports psychology',  
                    'competition mindset', 'pre-game rituals', 'goal setting', 'athleticism', 'sports tactics',  
                    'stadiums', 'sports broadcasting', 'spectators', 'fan engagement', 'match preparation',  
                    'warm-ups', 'cool-downs', 'recovery techniques', 'stretching routines', 'athletic wear',  
                    'conditioning', 'referees', 'officiating', 'sports analytics', 'scouting',  
                    'youth sports', 'adaptive sports', 'professional leagues', 'amateur competitions', 'medal ceremonies',  
                    'paralympics', 'olympics', 'sports history', 'rivalries', 'records and milestones',  
                    'sports ethics', 'team dynamics', 'fan culture', 'sports marketing', 'community leagues',  
                    'sports innovations', 'season schedules', 'sports agents', 'endorsements', 'team branding'  
                ],  
                'weight': 1.3  
            },
            'Culture' :{
                'keywords': [  
                    'traditions', 'customs', 'heritage', 'rituals', 'folklore',  
                    'language', 'dialects', 'literature', 'storytelling', 'music',  
                    'dance', 'festivals', 'celebrations', 'religion', 'philosophy',  
                    'artifacts', 'architecture', 'cuisine', 'ethnicity', 'identity',  
                    'multiculturalism', 'globalization', 'diversity', 'intercultural exchange', 'cultural preservation',  
                    'ancestry', 'mythology', 'symbols', 'values', 'beliefs',  
                    'ceremonies', 'cultural narratives', 'oral history', 'traditional crafts', 'indigenous practices',  
                    'urban culture', 'subcultures', 'pop culture', 'counterculture', 'cross-cultural communication',  
                    'cultural diplomacy', 'cultural heritage sites', 'language revitalization', 'music traditions', 'ritual dances',  
                    'heritage conservation', 'cultural festivals', 'folk music', 'regional foods', 'storytelling techniques',  
                    'migration patterns', 'cultural adaptation', 'anthropology', 'sociocultural studies', 'identity politics',  
                    'cultural artifacts', 'world heritage', 'cultural icons', 'oral traditions', 'native languages',  
                    'regional dialects', 'folkloric performances', 'cultural exchange programs', 'community traditions', 'heritage tours',  
                    'traditional attire', 'cultural ceremonies', 'cultural innovation', 'regional art styles', 'cultural diversity',  
                    'social norms', 'taboos', 'historical customs', 'cultural anthropology', 'traditional wisdom'  
                ],  
                'weight': 1.2  
            },
            'Travel' : {
                'keywords': [  
                    'tourism', 'destinations', 'adventure', 'exploration', 'itineraries',  
                    'vacations', 'hotels', 'airlines', 'backpacking', 'road trips',  
                    'travel blogs', 'landmarks', 'cultural experiences', 'beaches', 'mountains',  
                    'urban travel', 'rural escapes', 'eco-tourism', 'wildlife tours', 'guided tours',  
                    'travel photography', 'packing tips', 'passport', 'visa', 'local cuisine',  
                    'souvenirs', 'travel guides', 'hostels', 'luxury resorts', 'cruises',  
                    'budget travel', 'city tours', 'travel insurance', 'adventure sports', 'hiking trails',  
                    'scenic routes', 'cultural immersion', 'world exploration', 'remote destinations', 'national parks',  
                    'travel planning', 'road maps', 'transportation', 'travel safety', 'travel blogs',  
                    'holiday packages', 'hidden gems', 'off-the-beaten-path', 'travel hacks', 'seasonal travel',  
                    'vacation rentals', 'travel influencers', 'trip preparation', 'day trips', 'local transportation',  
                    'international travel', 'group tours', 'solo travel', 'travel companions', 'itinerary customization',  
                    'travel agencies', 'last-minute deals', 'frequent flyer programs', 'luggage recommendations', 'travel bucket lists',  
                    'adventurous getaways', 'relaxing retreats', 'family vacations', 'city breaks', 'travel reviews',  
                    'historical sites', 'eco-friendly travel', 'travel expenses', 'travel trends', 'photographic spots'  
                ],  
                'weight': 1.3  
            },
            'Entertainment' : {
                'keywords': [  
                    'movies', 'television', 'streaming', 'theater', 'cinema',  
                    'music', 'concerts', 'festivals', 'celebrities', 'pop culture',  
                    'video games', 'gaming consoles', 'eSports', 'podcasts', 'stand-up comedy',  
                    'improv', 'live performances', 'web series', 'reality shows', 'documentaries',  
                    'animated films', 'blockbusters', 'indie films', 'movie premieres', 'award shows',  
                    'red carpet', 'soundtracks', 'music videos', 'album releases', 'lyricists',  
                    'playwrights', 'screenwriters', 'set design', 'showrunners', 'binge-watching',  
                    'sound engineering', 'special effects', 'cinematography', 'director', 'producers',  
                    'acting', 'casting', 'scriptwriting', 'film festivals', 'box office',  
                    'merchandising', 'fan communities', 'fandoms', 'trivia', 'spoilers',  
                    'entertainment news', 'celebrity gossip', 'media criticism', 'cultural commentary', 'fan fiction',  
                    'streaming platforms', 'sound editing', 'voice acting', 'animated series', 'classic films',  
                    'film criticism', 'TV ratings', 'show reviews', 'trailers', 'behind the scenes',  
                    'cinematic universes', 'premiere events', 'digital entertainment', 'interactive storytelling', 'virtual concerts',  
                    'immersive experiences', 'entertainment law', 'studio productions', 'entertainment marketing', 'pop icons',  
                    'reboots', 'spin-offs', 'limited series', 'sound design', 'fan theories',  
                    'trending shows', 'cult classics', 'niche genres', 'entertainment franchises', 'on-screen chemistry'  
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
