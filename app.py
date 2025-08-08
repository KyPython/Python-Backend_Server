from flask import Flask, request, jsonify
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

app = Flask(__name__)

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ✅ ADD ROOT ROUTE - This is what's missing!
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SmartScan AI Python Backend is running!",
        "status": "healthy",
        "service": "python-backend",
        "endpoints": {
            "/": "GET - Health check",
            "/health": "GET - Health status", 
            "/local-caption": "POST - Generate image captions"
        },
        "model": "Salesforce/blip-image-captioning-base",
        "version": "1.0.0"
    })

# ✅ ADD HEALTH CHECK
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy", 
        "service": "python-backend",
        "model_loaded": True
    })

@app.route("/local-caption", methods=["POST"])
def local_caption():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"output": "No JSON data provided"}), 400
            
        image_url = data.get("imageUrl")
        if not image_url:
            return jsonify({"output": "No imageUrl provided"}), 400

        print(f"Processing image: {image_url}")

        # Download and process the image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Generate caption using BLIP model
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        print(f"Generated caption: {caption}")
        
        return jsonify({"output": caption})
        
    except requests.exceptions.RequestException as e:
        print(f"Image download error: {str(e)}")
        return jsonify({"output": f"Failed to download image: {str(e)}"}), 400
    except Exception as e:
        print(f"Error in /local-caption: {str(e)}")
        return jsonify({"output": f"Error processing image: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)