# Import necessary libraries for the Flask web server and image processing
from flask import Flask, request, jsonify  # Web server framework and utilities
import joblib  # For loading pre-trained machine learning models
import torch  # PyTorch deep learning framework
from torchvision import transforms  # Image transformations for neural networks
from PIL import Image, ImageFilter  # Python Imaging Library for image manipulation
import timm  # PyTorch Image Models library for pre-trained models
import logging  # Logging utilities for server diagnostics
import time  # Time utilities for performance measurement
import os  # Operating system utilities for file paths

# === Setup logging ===
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')  # Create logs folder in the same directory
os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist
log_file = os.path.join(log_dir, 'server_log.txt')  # Path to the log file

# Configure the logging system with both file and console output
logging.basicConfig(
    level=logging.INFO,  # Set minimum logging level to INFO (excludes DEBUG messages)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Timestamp, module, level, and message format
    handlers=[
        logging.FileHandler(log_file),  # Write logs to file
        logging.StreamHandler()  # Also output to console
    ]
)
logger = logging.getLogger(__name__)  # Get logger for this file
logger.info(f"Logging to: {log_file}")  # First log entry with log file location

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
logger.info(f"Using device: {device}")  # Log which device is being used

# === Load Models ===
# First model: Image classifier to determine if the image contains chicken
mini_identifier_model = timm.create_model('deit_tiny_distilled_patch16_224', num_classes=2)  # Vision transformer with 2 output classes
mini_identifier_model.load_state_dict(torch.load("deit_tiny_chicken_classifier.pth", map_location=device))  # Load pre-trained weights
mini_identifier_model.to(device)  # Move model to the appropriate device (GPU/CPU)
mini_identifier_model.eval()  # Set model to evaluation mode (disables dropout, etc.)

# Second model: SVM classifier for consumability assessment
svm_model = joblib.load("svm_model_20250427_194931.pkl")  # Load the Support Vector Machine model

# Feature extractor: Pre-trained vision transformer for feature extraction
deit = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=True)  # Use pre-trained weights
deit.reset_classifier(0)  # Remove the classification head, keeping only feature extraction
deit.to(device)  # Move model to the appropriate device
deit.eval()  # Set model to evaluation mode

# === Image Transform ===
# Define the image preprocessing pipeline for the neural network
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (required input size)
    transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor (values between 0-1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'jpg','png'}

def allowed_file(filename):
    """
    Check if the file has an allowed extension (.jpg or .png)
    
    Args:
        filename: The name of the uploaded file
        
    Returns:
        bool: True if the file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_blurry(image_pil, threshold=100):
    """
    Detect if an image is too blurry based on Laplacian variance.
    Laplacian filter detects edges, and the variance of the resulting image
    indicates the amount of edge detail. Low variance means few edges (blurry).
    
    Args:
        image_pil: PIL Image to check for blurriness
        threshold: Variance threshold below which an image is considered blurry
        
    Returns:
        bool: True if the image is blurry, False otherwise
    """
    image_gray = image_pil.convert('L')  # Convert to grayscale for edge detection
    image_laplacian = image_gray.filter(ImageFilter.FIND_EDGES)  # Apply edge detection filter
    
    # Fix: Convert the PIL Image to numpy array first, then calculate variance
    import numpy as np
    laplacian_array = np.array(image_laplacian)
    variance = np.var(laplacian_array)  # Calculate variance of edge-detected image
    
    logger.debug(f"Blurriness variance: {variance}")  # Log blurriness score
    return variance < threshold  # Return True if below threshold (blurry)

def is_small_image(image_pil, min_width=100, min_height=100):
    """
    Check if an image is too small for accurate analysis
    
    Args:
        image_pil: PIL Image to check dimensions
        min_width: Minimum acceptable width in pixels
        min_height: Minimum acceptable height in pixels
        
    Returns:
        bool: True if the image is smaller than the minimum dimensions
    """
    width, height = image_pil.size  # Get image dimensions
    logger.debug(f"Image size: {width}x{height}")
    return width < min_width or height < min_height  # True if either dimension is too small

# === Flask App ===
app = Flask(__name__)  # Create the Flask application

@app.route('/predict', methods=['POST'])  # Define API endpoint that accepts POST requests
def predict():
    """
    Process an image to determine chicken breast consumability.
    Workflow:
    1. Validate the uploaded file
    2. Check if the image is suitable for processing (not blurry, not too small)
    3. Use the identifier model to check if it contains chicken breast
    4. Extract features using the DeiT model
    5. Use the SVM model to predict consumability class
    
    Returns:
        JSON response with prediction or error message
    """
    start_time = time.time()  # Start timing the request for performance monitoring
    logger.info("Received new prediction request.")

    # Check if the request contains an image file
    if 'image' not in request.files:
        logger.error("No image part in the request")
        return jsonify({'error': 'No image uploaded'}), 400  # HTTP 400 Bad Request

    file = request.files['image']  # Get the uploaded file

    # Validate file extension
    if not allowed_file(file.filename):
        logger.error(f"Disallowed file extension: {file.filename}")
        return jsonify({'error': 'File type not allowed. Please upload a .jpg or .png file.'}), 400

    try:
        # Open and convert the image to RGB format (necessary for model processing)
        image = Image.open(file).convert("RGB")

        # Check if image is too blurry for accurate analysis
        if is_blurry(image):
            logger.warning("Uploaded image is too blurry.")
            return jsonify({'error': 'Uploaded image is too blurry. Please upload a clearer image.'}), 400

        # Check if image dimensions are sufficient
        if is_small_image(image):
            logger.warning("Uploaded image is too small.")
            return jsonify({'error': 'Uploaded image is too small. Please upload a larger image.'}), 400

        # Preprocess the image for the neural network
        img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Step 1: Use the identifier model to verify this is chicken breast
        with torch.no_grad():  # Disable gradient calculations for inference
            outputs = mini_identifier_model(img_tensor)  # Forward pass through the model
            probs = torch.softmax(outputs, dim=1)  # Convert outputs to probabilities
            confidence_score = probs[0][1].item()  # Get confidence score for "chicken breast" class
            is_chicken_breast = torch.argmax(outputs, dim=1).item() == 1  # Check if predicted class is chicken breast

        # Reject the image if it's not confidently recognized as chicken breast
        if not is_chicken_breast or confidence_score < 0.8:  # 0.8 = 80% confidence threshold
            logger.warning(f"Rejected non-chicken image (confidence: {confidence_score:.2f})")
            return jsonify({'error': f'The uploaded image is not confidently recognized as a raw chicken breast (confidence: {confidence_score:.2f}). Please upload a valid image.'}), 400

        # Step 2: Extract features using the DeiT model for SVM classification
        with torch.no_grad():
            features = deit.forward_features(img_tensor)  # Extract features without classification
            feature_vector = features.mean(dim=1).cpu().numpy()  # Global average pooling & convert to numpy

        # Step 3: Use SVM model to classify consumability based on the extracted features
        prediction = svm_model.predict(feature_vector)
        labels = ["Consumable", "Half-consumable", "Not consumable"]  # Class labels
        label = labels[int(prediction[0])]  # Get the predicted label

        # Calculate total processing time
        processing_time = time.time() - start_time
        logger.info(f"Prediction complete: {label} (confidence: {confidence_score:.2f}), time: {processing_time:.3f}s")

        # Prepare response with prediction results and processing time
        response = {
            'prediction': label,  # The consumability classification
            'confidence': str(round(confidence_score, 2)),  # Confidence score (0-1)
            'processing_time_sec': processing_time  # Time taken for processing
        }

        return jsonify(response)  # Return the prediction result as JSON

    except Exception as e:
        # Log and handle any errors during processing
        processing_time = time.time() - start_time
        logger.error(f"Exception during prediction ({processing_time:.3f}s): {str(e)}")
        return jsonify({'error': str(e)}), 500  # HTTP 500 Internal Server Error

if __name__ == "__main__":
    try:
        # Start the Flask server on port 5000 (or from environment variable)
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
    except OSError as e:
        logger.error(f"Error starting app: {e}")
        exit(1)  # Exit with error code if server fails to start
