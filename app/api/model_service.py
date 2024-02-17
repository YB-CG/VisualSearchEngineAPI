# app/api/model_service.py

import os
import pickle
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import pandas as pd
import requests
import json
from werkzeug.utils import secure_filename


# Define relative paths for your image directories
cutout_img_dir = "data/cutout-img/cutout"
model_img_dir = "data/model-img/model"

# Load saved features from file
with open('models/image_features_vgg.pkl', 'rb') as f:
    loaded_features = pickle.load(f)

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract_features(self, img):
        img = img.resize((224, 224))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = vgg_preprocess(x)
        features = self.model.predict(x)
        features = features / np.linalg.norm(features)
        return features

# Instantiate FeatureExtractor
vgg_feature_extractor = FeatureExtractor()

def load_and_process_data():
    # Read the data files
    listing_data = pd.read_csv("data/current_farfetch_listings.csv")
    # drop the unnamed: 0 column
    listing_data.drop('Unnamed: 0', axis=1, inplace=True)
    # Drop priceInfo.installmentsLabel
    listing_data.drop('priceInfo.installmentsLabel', axis=1, inplace=True)
    # Drop the column merchandiseLabel
    listing_data.drop('merchandiseLabel', axis=1, inplace=True)
    # fill the null values in priceInfo.discountLabel with 0
    listing_data['priceInfo.discountLabel'] = listing_data['priceInfo.discountLabel'].fillna(0)

    # list the directories
    cutout_images = os.listdir(cutout_img_dir)
    model_images = os.listdir(model_img_dir)

    # Function to extract image name from path
    def extractImageName(x):
        return os.path.basename(x)

    listing_data['cutOutimageNames'] = listing_data['images.cutOut'].apply(lambda x: extractImageName(x))
    listing_data['modelimageNames'] = listing_data['images.model'].apply(lambda x: extractImageName(x))

    # Extract only those data points for which we have images
    listing_data = listing_data[listing_data['cutOutimageNames'].isin(cutout_images)]
    listing_data = listing_data[listing_data['modelimageNames'].isin(model_images)]
    # Reset the index
    listing_data.reset_index(drop=True, inplace=True)
    # Add entire paths to cutOut and modelImages
    listing_data['cutOutImages_path'] = listing_data['cutOutimageNames'].apply(lambda x: os.path.join(cutout_img_dir, x))
    listing_data['modelImages_path'] = listing_data['modelimageNames'].apply(lambda x: os.path.join(model_img_dir, x))

    # listing_data['cutOutImages_path'] = os.path.join(cutout_img_dir, listing_data['cutOutimageNames'])
    # listing_data['modelImages_path'] = os.path.join(model_img_dir, listing_data['modelimageNames'])
    # Drop the cutOutimageNames, cutOutimageNames
    listing_data.drop(['cutOutimageNames', 'cutOutimageNames'], axis=1, inplace=True)

    return listing_data
# def process_image(image_file, url, listing_data):
def process_image(image_file=None, image_url=None, listing_data=None):
    try:
        if image_file:
            # Save the uploaded image to a temporary location
            filename = secure_filename(image_file.filename)
            temp_path = os.path.join('/home/skay/Documents/Project/Capstone/VIsual-Search-Engine/API/temp', filename)
            image_file.save(temp_path)
            query_img = Image.open(temp_path)
        elif image_url:
            # Load the image from the URL
            response = requests.get(image_url)
            response.raise_for_status()
            query_img = Image.open(BytesIO(response.content))
        else:
            return json.dumps({'error': 'No image file or URL provided'})

        # Extract features from the query image using the loaded feature extractor
        query_features = vgg_feature_extractor.extract_features(query_img)

        # Compute similarity using Euclidean Distance for the loaded model
        similarity_images = {}
        for idx, feat in loaded_features.items():
            similarity_images[idx] = np.sum((query_features - feat)**2) ** 0.5

        # Extract the top 10 similar images
        similarity_sorted = sorted(similarity_images.items(), key=lambda x: x[1])
        top_12_indexes = [idx for idx, _ in similarity_sorted[:12]]

        # Extracting additional information

        top_12_similar_imgs = listing_data.iloc[top_12_indexes]['modelImages_path']
        top_12_similar_cutOut_imgs = listing_data.iloc[top_12_indexes]['cutOutImages_path']
        brand_names = listing_data.iloc[top_12_indexes]['brand.name']
        prices = listing_data.iloc[top_12_indexes]['priceInfo.finalPrice']
        short_descriptions = listing_data.iloc[top_12_indexes]['shortDescription']
        stock_totals = listing_data.iloc[top_12_indexes]['stockTotal'].astype(int).tolist()

        # Prepare the result as a dictionary
        result = {

            'similar_images': [
                {
                    'id': int(idx),
                    'img_path': img_path,
                    'Detail_img_path': Detail_img_path,
                    'brand': brand,
                    'price': price,
                    'shortDescription': short_description,
                    'stockTotal': stock_total
                }
                for idx, img_path, Detail_img_path, brand, price, short_description, stock_total in zip(
                    top_12_indexes, top_12_similar_cutOut_imgs, top_12_similar_imgs, brand_names, prices, short_descriptions, stock_totals
                )
            ]
        }

        # Serialize the result dictionary to JSON format
        json_result = json.dumps(result)

        return json_result

    except Image.DecompressionBombError as e:
        # You might want to return a JSON error message here as well
        return json.dumps({'error': f"DecompressionBombError for the query image: {e}"})
    except UnidentifiedImageError as e:
        return json.dumps({'error': f"UnidentifiedImageError for the query image: {e}"})
    except requests.RequestException as e:
        return json.dumps({'error': f"Error fetching image from URL: {e}"})
    except Exception as e:
        return json.dumps({'error': str(e)})
