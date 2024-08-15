import os
from flask import Flask, request, jsonify
import google.generativeai as genai

# Configuration de l'API Google AI
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = Flask(__name__)

# Configuration du modèle
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

@app.route('/api/gemini_vision', methods=['POST', 'GET'])
def gemini_vision():
    if request.method == 'POST':
        # POST request: expects 'text' and 'image' in the form-data
        text = request.form.get('text')
        image = request.files.get('image')
        
        if image:
            mime_type = image.mimetype
            image_path = os.path.join("/tmp", image.filename)
            image.save(image_path)
            uploaded_file = upload_to_gemini(image_path, mime_type)
            
            chat_history = [
                {
                    "role": "user",
                    "parts": [
                        uploaded_file,
                        text,
                    ],
                }
            ]
            
            chat_session = model.start_chat(history=chat_history)
            response = chat_session.send_message(text)
            return jsonify({'response': response.text})
        else:
            return jsonify({'error': 'Image file not provided'}), 400

    elif request.method == 'GET':
        # GET request: expects 'text' as a query parameter
        text = request.args.get('text')
        if text:
            chat_history = [
                {
                    "role": "user",
                    "parts": [text],
                }
            ]
            
            chat_session = model.start_chat(history=chat_history)
            response = chat_session.send_message(text)
            return jsonify({'response': response.text})
        else:
            return jsonify({'error': 'Text parameter not provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
