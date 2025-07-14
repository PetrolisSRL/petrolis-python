#!/usr/bin/env python3
"""
Document Processor Migliorato
Integra gli algoritmi avanzati testati in test_simple.py
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.filters import threshold_otsu
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedDocumentProcessor:
    def __init__(self):
        self.min_area_ratio = 0.05

    def process_document_image(self, image_data: str, enhance: bool = True):
        """
        Processa un'immagine (base64) usando l'algoritmo basato su colore,
        la raddrizza e la restituisce come stringa base64.
        """
        try:
            logger.info("Starting image processing...")
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open image from base64: {e}")
            return None

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. Color-based segmentation (HSV)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 2. Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. Find contours on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"Found {len(contours)} contours on the mask.")

        if not contours:
            logger.warning("No contours found on mask, returning original image.")
            return self.image_to_base64(image)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        document_contour = None

        for contour in contours[:5]:
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.035 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                document_contour = approx
                logger.info(f"Found a 4-sided contour with area: {cv2.contourArea(contour)}")
                break
        
        if document_contour is None:
            logger.warning("No perfect quadrilateral found, using largest contour as fallback.")
            document_contour = contours[0]
        
        # 4. Apply perspective transform
        try:
            if len(document_contour.reshape(-1, 2)) == 4:
                points = document_contour
            else:
                rect = cv2.minAreaRect(document_contour)
                points = cv2.boxPoints(rect)

            result = self.apply_perspective_transform(image, points)
            logger.info(f"Perspective transform successful. New size: {result.size}")
        except Exception as e:
            logger.error(f"Perspective transform failed: {e}. Cropping with bounding box as fallback.")
            x, y, w, h = cv2.boundingRect(document_contour)
            cropped_cv = cv_image[y:y+h, x:x+w]
            cropped_rgb = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(cropped_rgb)

        # 5. Enhance image if requested
        if enhance:
            result = self.enhance_image(result)
            logger.info("Image enhancement applied.")

        return self.image_to_base64(result)

    def apply_perspective_transform(self, image, points):
        """Applica trasformazione prospettica con ordinamento robusto dei punti."""
        if len(points.shape) == 3:
            points = points.reshape(4, 2)
        
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)] # top-left
        rect[2] = points[np.argmax(s)] # bottom-right
        
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)] # top-right
        rect[3] = points[np.argmax(diff)] # bottom-left
        
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(cv_image, M, (maxWidth, maxHeight))
        
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(warped_rgb)

    def enhance_image(self, image):
        # Enhanced contrast and sharpness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        return image

    def image_to_base64(self, image):
        """Converte immagine PIL in base64"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

# Istanza globale del processor migliorato
improved_processor = ImprovedDocumentProcessor()

# Tutti gli endpoint rimangono identici, cambia solo l'algoritmo interno
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'improved-document-processor'})

@app.route('/process-document', methods=['POST'])
def process_document():
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image']
        enhance = data.get('enhance', True)
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        processed_image = improved_processor.process_document_image(image_data, enhance)
        
        return jsonify({
            'processed_image': processed_image,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Error in process_document endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to process document',
            'details': str(e)
        }), 500

@app.route('/process-document-url', methods=['POST'])
def process_document_from_url():
    try:
        data = request.get_json()
        
        if not data or 'image_url' not in data:
            return jsonify({'error': 'No image URL provided'}), 400
        
        import requests
        
        response = requests.get(data['image_url'])
        response.raise_for_status()
        
        image_base64 = base64.b64encode(response.content).decode()
        
        enhance = data.get('enhance', True)
        processed_image = improved_processor.process_document_image(image_base64, enhance)
        
        return jsonify({
            'processed_image': processed_image,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Error in process_document_from_url endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to process document from URL',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Improved Document Processor Microservice...")
    app.run(host='0.0.0.0', port=5001, debug=False) 