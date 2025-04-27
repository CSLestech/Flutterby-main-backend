from flask import Flask, request, jsonify
import joblib
import torch
from torchvision import transforms
from PIL import Image
import timm
import logging
import time
import os

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'server_log.txt')

# Set up logging for production with absolute path
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

# === 1. Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# === 2. Load Mini Identifier Model ===
# Use timm to load the classifier model with the same architecture as the training code
mini_identifier_model = timm.create_model('deit_base_distilled_patch16_224', num_classes=2)
mini_identifier_model.load_state_dict(torch.load("deit_chicken_classifier.pth", map_location=device))
mini_identifier_model.to(device)
mini_identifier_model.eval()

# === 3. Load SVM Model ===
svm_model = joblib.load("svm_model_20250420_102634.pkl")

# === 4. Load DeiT Feature Extractor ===
deit = timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
deit.reset_classifier(0)  # Remove the classification head
deit.eval()
deit.to(device)

# === 5. Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === 6. Flask App ===
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Start timing the prediction
    start_time = time.time()
    logger.info("Starting prediction request")
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image = Image.open(request.files['image']).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Step 1: Check if the image is a chicken breast
        with torch.no_grad():
            outputs = mini_identifier_model(img_tensor)
            logger.debug(f"Model Outputs: {outputs}")
            is_chicken_breast = torch.argmax(outputs, dim=1).item() == 1

        if not is_chicken_breast:
            return jsonify({'error': 'The uploaded image is not a raw chicken breast. Please upload a valid image.'}), 400

        # Step 2: Proceed with the main algorithm (feature extraction)
        with torch.no_grad():
            features = deit.forward_features(img_tensor)
            feature_vector = features.mean(dim=1).cpu().numpy()

        # Step 3: Predict using the SVM model
        prediction = svm_model.predict(feature_vector)
        labels = ["Consumable", "Half-consumable", "Not consumable"]
        label = labels[int(prediction[0])]
        
        # Calculate confidence (this is a placeholder - in a real app, you'd get this from the model)
        confidence = 0.9  # Example confidence value
        
        # Log performance timing for server-side processing
        prediction_time = time.time() - start_time
        logger.info(f"Server prediction completed in {prediction_time:.3f} seconds")
        
        # Include timing information in response
        response = {
            'prediction': label,
            'confidence': str(confidence),
            'processing_time_sec': prediction_time
        }
        
        # Log if prediction exceeds target time
        if prediction_time > 3.0:
            logger.warning(f"⚠️ Prediction took {prediction_time:.3f} seconds, exceeding 3-second threshold")
        
        return jsonify(response)
        
    except Exception as e:
        # Log the error
        prediction_time = time.time() - start_time
        logger.error(f"Error in prediction after {prediction_time:.3f} seconds: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
