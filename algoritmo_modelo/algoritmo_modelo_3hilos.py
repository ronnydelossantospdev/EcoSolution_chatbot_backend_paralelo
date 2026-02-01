import pandas as pd
import time;
import re
import threading
import joblib
import queue
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def limpiar_texto(texto):
    texto = texto.lower()  
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto) 
    return texto

def cacular_tiempo(nombre,incio, fin):
    print(f"Tiempo de {nombre}: {round(fin - incio, 3)}")


# Cargar el dataset
def cargarDataset():
    df = pd.read_csv('dataset/medio_ambiente_dataset.csv' )
    return df


# Limpiar los datos
def limpiar_datos_pregunta(df):
    dfLimpio = df.copy() 
    dfLimpio['Preguntas'] = dfLimpio['Preguntas'].apply(limpiar_texto)
    dfLimpio['Respuestas'] = dfLimpio['Respuestas'].apply(limpiar_texto)
    return dfLimpio


# Separar características (Pregunta) y etiquetas (Respuestas)
def separarColumnas(dfLimpio):
    x = dfLimpio['Preguntas']
    y = dfLimpio['Respuestas']
    return x, y


# Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_entrenamiento_prueba(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Vectorización del texto usando TF-IDF
def vectorizar_texto(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer


# Crear y entrenar el modelo de regresión logística
def entrenar_modelo(X_train_tfidf, y_train):
    modelo = LogisticRegression(random_state=42, max_iter=1000)
    modelo.fit(X_train_tfidf, y_train)
    return modelo


# Evaluar el modelo
def evaluar_modelo(modelo, X_test_tfidf, y_test):
    y_pred = modelo.predict(X_test_tfidf)
    precision = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {precision:.2%}')
    print(classification_report(y_test, y_pred))


# Función para interactuar con el chatbot
def chatbot(pregunta, vectorizer, modelo):
    pregunta_limpia = limpiar_texto(pregunta)
    pregunta_tfidf = vectorizer.transform([pregunta_limpia])
    respuesta = modelo.predict(pregunta_tfidf)[0]
    return respuesta


# Interactuar con el modelo
def intereactuar(vectorizer, modelo):
    print("\n¡El chatbot está listo para responder tus preguntas sobre medio ambiente!")
    while True:
        pregunta = input("Haz una pregunta (o escribe 'salir' para terminar): ")
        if pregunta.lower() == 'salir':
            print("¡Hasta luego!")
            break
        respuesta_chatbot = chatbot(pregunta, vectorizer, modelo)
        print(f"Respuesta: {respuesta_chatbot}")


# Guardar el Modelo y el Vectorizador
def guardar_modelo(modelo,vectorizer):
   joblib.dump(modelo, 'chatbot/modelo/modelo.pkl') # Guardar el modelo de regresión logística
   joblib.dump(vectorizer, 'chatbot/modelo/vectorizer_tfidf.pkl') # Guardar el vectorizador TF-IDF
   print("Modelo y vectorizador guardados exitosamente.")


def hilo_flujo(vectorizer_cola_hilo, parametros_modelos):

    # Cargar y limpiar datos
    df = cargarDataset()
    dfLimpio = limpiar_datos_pregunta(df)

    # Separar columnas
    x, y = separarColumnas(dfLimpio)

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = dividir_entrenamiento_prueba(x, y)

    # Vectorizar texto
    X_train_tfidf, X_test_tfidf, vectorizer = vectorizar_texto(X_train, X_test)

    # Entrenar el modelo
    modelo = entrenar_modelo(X_train_tfidf, y_train)

    # Evaluar el modelo
    #evaluar_modelo(modelo, X_test_tfidf, y_test)
    
    # Hacer preguntas
    #intereactuar(vectorizer, modelo)

    # Poner el modelo y el vectorizador en la cola para ser usados en otro hilo
    vectorizer_cola_hilo.put(vectorizer)
    parametros_modelos.put((X_train_tfidf, y_train))
    


# Hilo guardar modelo
def hilo_guardar():
    vectorizer = vectorizer_cola_hilo.get()
    modelo = modelo_cola_hilo.get()
    guardar_modelo(modelo, vectorizer)



# Hilo entrenar modelo
def hilo_modelo(modelo_cola_hilo):

    X_train_tfidf, y_train = parametros_modelos.get()

    # Entrenar el modelo
    modelo = entrenar_modelo(X_train_tfidf, y_train)
    modelo_cola_hilo.put(modelo)



vectorizer_cola_hilo = queue.Queue()  # Cola de hilo vectorizer
modelo_cola_hilo = queue.Queue() # Cola de hilo modelo
parametros_modelos = queue.Queue() # Cola de hilo de parametros modelo

hilo_flujo_datos = threading.Thread(target=hilo_flujo, args=(vectorizer_cola_hilo, parametros_modelos))  # Hilo del flijo de datos
hilo_entrenar_modelo = threading.Thread(target=hilo_modelo, args=(modelo_cola_hilo,)) # Hilo entrenar modelo
hilo_guardar_modelo = threading.Thread(target=hilo_guardar)  # Hilo guardar modelo



inicio = time.time() # inicio tiempo

i_hf =  time.time() # inicio tiempo hilo 1
hilo_flujo_datos.start()
hilo_flujo_datos.join()
f_hf = time.time() # fin tiempo hilo 1

i_he =  time.time() # inicio tiempo hilo 2
hilo_entrenar_modelo.start()
hilo_entrenar_modelo.join()
f_he = time.time() # fin tiempo hilo 2

i_hg =  time.time() # inicio tiempo hilo 3
hilo_guardar_modelo.start()
hilo_guardar_modelo.join()
f_hg = time.time() # fin tiempo hilo 3

fin = time.time() # fin tiempo

cacular_tiempo("Tiempo de ejecaucion del hilo_flujo_datos",i_hf,f_hf)
cacular_tiempo("Tiempo de ejecaucion del hilo_flujo_datos",i_he,f_he)
cacular_tiempo("Tiempo de ejecaucion del hilo_guardar_modelo",i_hg,f_hg)
cacular_tiempo("Tiempo de ejecaucion de los 3 hilos",inicio,fin)



