import os
from flask import Flask, request, jsonify
import google.generativeai as genai
import uuid  # Pour générer des identifiants uniques

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

# Dictionnaire global pour stocker l'historique de chaque utilisateur
user_histories = {}
# Compteur pour générer des user_id simples
user_counter = 0

def generate_user_id():
    """Génère un identifiant utilisateur séquentiel."""
    global user_counter
    user_counter += 1
    return str(user_counter)  # Les IDs générés seront '1', '2', '3', etc.

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

@app.route('/api/gemini_vision', methods=['GET'])
def gemini_vision_get():
    user_id = request.args.get('user_id')
    text = request.args.get('text')
    image_url = request.args.get('image_url')

    # Si aucun user_id n'est fourni, en générer un
    if not user_id:
        user_id = generate_user_id()

    # Récupérer l'historique de l'utilisateur ou en créer un nouveau
    if user_id not in user_histories:
        user_histories[user_id] = []

    user_history = user_histories[user_id]

    if image_url and text:
        # Simuler le téléchargement de l'image (remplacer par une méthode pour gérer les images si nécessaire)
        image_path = download_image(image_url)
        uploaded_file = upload_to_gemini(image_path, mime_type="image/jpeg")

        user_history.append({
            "role": "user",
            "parts": [uploaded_file, text],
        })

        chat_session = model.start_chat(history=user_history)
        response = chat_session.send_message(text)
        user_history.append({"role": "model", "parts": [response.text]})
        return jsonify({'user_id': user_id, 'response': response.text})

    elif text:
        user_history.append({
            "role": "user",
            "parts": [text],
        })

        chat_session = model.start_chat(history=user_history)
        response = chat_session.send_message(text)
        user_history.append({"role": "model", "parts": [response.text]})
        return jsonify({'user_id': user_id, 'response': response.text})

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
