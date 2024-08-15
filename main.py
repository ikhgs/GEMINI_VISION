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

@app.route('/api/gemini_vision', methods=['GET'])
def gemini_vision_get():
    # GET request: expects 'text' and 'image_url' as query parameters
    text = request.args.get('text')
    image_url = request.args.get('image_url')

    if image_url and text:
        # Simuler le téléchargement de l'image (remplacer par une méthode pour gérer les images si nécessaire)
        image_path = download_image(image_url)
        uploaded_file = upload_to_gemini(image_path, mime_type="image/jpeg")

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
    elif text:
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
        return jsonify({'error': 'Text or image_url parameter not provided'}), 400

def download_image(image_url):
    """Télécharge une image depuis une URL et retourne le chemin local."""
    import requests
    from urllib.parse import urlparse
    import os

    # Crée un chemin temporaire pour enregistrer l'image
    response = requests.get(image_url, stream=True)
    image_path = os.path.join("/tmp", os.path.basename(urlparse(image_url).path))
    
    with open(image_path, 'wb') as out_file:
        out_file.write(response.content)
    
    return image_path

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
