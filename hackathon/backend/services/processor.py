import PyPDF2
from io import BytesIO

class Processor:
    def extract_text(self, file_object):
       
        try:
            # Save the current position of file pointer
            if hasattr(file_object, 'seek'):
                file_object.seek(0)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(file_object)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error extracting text: {str(e)}")