# Cargar el Modelo y el Vectorizador
# En un archivo separado o cuando desees utilizar el chatbot, puedes cargar el modelo y el vectorizador de la siguiente manera:

import re
import joblib

# Cargar el modelo y el vectorizador
modelo = joblib.load('chatbot/modelo/modelo.pkl')
vectorizer = joblib.load('chatbot/modelo/vectorizer_tfidf.pkl')

# Función para limpiar el texto (minúsculas y eliminación de caracteres especiales)
def limpiar_texto(texto):
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto)  # Eliminar caracteres especiales
    return texto


# Función para interactuar con el chatbot
def chatbot(pregunta):
    pregunta_limpia = limpiar_texto(pregunta)
    pregunta_tfidf = vectorizer.transform([pregunta_limpia])
    respuesta = modelo.predict(pregunta_tfidf)[0]
    return respuesta


# Ejemplo de uso
# pregunta = "¿Qué es el cambio climático?"
# respuesta = chatbot(pregunta)
# print(f"Respuesta: {respuesta}")


# # Ejemplo de interacción
# print("\n¡El chatbot está listo para responder tus preguntas sobre medio ambiente!")
# while True:
#     pregunta_usuario = input("Haz una pregunta (o escribe 'salir' para terminar): ")
#     if pregunta_usuario.lower() == 'salir':
#         print("¡Hasta luego!")
#         break
#     respuesta_chatbot = chatbot(pregunta_usuario)
#     print(f"Respuesta: {respuesta_chatbot}")