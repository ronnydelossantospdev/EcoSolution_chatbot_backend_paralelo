from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

# Ruta de bienvenida
@app.route('/')
def home():
    return jsonify({'mensaje': ['¡Hola!', '¿Cómo puedo ayudarte?']})

# Ruta POST para enviar datos
@app.route('/datos', methods=['POST'])
def set_informacion():
    modelo = joblib.load('./chatbot/modelo/modelo.pkl')
    vectorizer = joblib.load('./chatbot/modelo/vectorizer_tfidf.pkl')

    pregunta = request.json.get('pregunta')  # Recibe el JSON enviado y extrae la pregunta
    if not pregunta:
        return jsonify({'error': 'No se recibió una pregunta válida'}), 400

    pregunta = pregunta.lower() 
    pregunta_limpia = re.sub(r'[^a-záéíóúñü\s]', '', pregunta)

    pregunta_tfidf = vectorizer.transform([pregunta_limpia])
    respuesta = modelo.predict(pregunta_tfidf)[0]

    return jsonify({'mensaje': respuesta}), 201

if __name__ == '__main__':
    app.run(debug=True)
