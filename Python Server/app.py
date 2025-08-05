from flask import Flask, request, jsonify
from PIL import Image
import requests
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/local-caption", methods=["POST"])
def local_caption():
    data = request.get_json()
    image_url = data.get("imageUrl")
    if not image_url:
        return jsonify({"output": "No imageUrl provided"}), 400

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return jsonify({"output": caption})
    except Exception as e:
        print("Error in /local-caption:", str(e))
        return jsonify({"output": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=5001)