# app/api/routes.py

from flask import Blueprint, request, jsonify
from flasgger import swag_from
from .model_service import process_image, load_and_process_data

api_bp = Blueprint('api', __name__)

listing_data = load_and_process_data()

@api_bp.route('/predict', methods=['POST'])
@swag_from('/home/skay/Documents/Project/Capstone/VIsual-Search-Engine/API/swagger.yml')  # Point to the Swagger YAML file for this route
def predict():
    """Endpoint to predict similar images.

    ---
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: The image file to process.
      - name: url
        in: formData
        type: string
        required: false
        description: The URL of the image to process.
    responses:
      200:
        description: The prediction result.
      400:
        description: Bad request.
      500:
        description: Internal server error.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        # url = request.form.get('url')

        # result = process_image(image_file, url, listing_data)
        result = process_image(image_file, listing_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
