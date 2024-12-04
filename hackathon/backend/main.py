from flask import Flask, request, jsonify, send_from_directory
from services.processor import Processor
from services.classifier import Classifier
from flask_cors import CORS 
import os
import logging
app = Flask(__name__)
processor = Processor()
classifier = Classifier()
last_uploaded_file = None
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("Received upload request")
        logger.debug(f"Request files: {request.files}")
        
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        logger.info(f"Received file: {file.filename}")
        
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if not file.filename.endswith('.pdf'):
            logger.warning("Invalid file type")
            return jsonify({'error': 'File must be PDF'}), 400
            
        content = file.read()
        logger.info(f"Successfully read file of size: {len(content)} bytes")
        
        return jsonify({
            'success': True,
            'message': 'File received successfully',
            'filename': file.filename,
            'size': len(content)
        })
        
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("Starting server...")
    app.run(debug=True, port=5000)