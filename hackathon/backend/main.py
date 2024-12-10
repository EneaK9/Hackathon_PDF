from flask import Flask, request, jsonify, send_from_directory
from services.processor import Processor
from services.classifier import Classifier
from flask_cors import CORS 
import os
import logging

app = Flask(__name__)
CORS(app)
processor = Processor()
classifier = Classifier()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.endswith('.pdf'):
            return jsonify({'error': 'File must be PDF'}), 400
        
        # Extract text from PDF
        text = processor.extract_text(file)
        
        # Classify the document
        category, confidence = classifier.classify_document(text)
        
        # Get detailed analysis if needed
        analysis = classifier.get_detailed_analysis(text)
        
        return jsonify({
            'category': category,
            'confidence': confidence,
            'analysis': analysis,
            'filename': file.filename
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting server...")
    app.run(debug=True, port=5000)