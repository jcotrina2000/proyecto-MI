from functools import lru_cache
import pandas as pd
import numpy as np
import boto3
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from datetime import timedelta
import pickle
import requests
import os
from flask_socketio import SocketIO, emit
import uuid
import time

app = Flask(__name__)
results_store = {}  # Diccionario para almacenar los resultados por sesión

def leer_datos(ruta):
    return pd.read_csv(ruta, delimiter=';')


def validar_y_actualizar(fila):
    if  fila["cantidad_unid"] >= 1:
        if fila['id_item'] == 13887:  # JERINGA MEGA INSUL 1MLx29Gx1/2x100
            fila["cantidad_frac"] += 100 * int(fila["cantidad_unid"])
        elif fila['id_item'] in {90765, 79680, 27112, 1669, 101609}:
            fila["cantidad_frac"] += int(fila["cantidad_unid"])
        elif fila['id_item'] == 54122:  # XARELTO COM-RECx10MGx10
            fila["cantidad_frac"] += 10 * int(fila["cantidad_unid"])
        elif(fila['id_item'] == 88275): #MICARDIX
            fila['cantidad_frac'] += 28 * int(fila["cantidad_unid"])
    return fila

def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.square(y_pred - y_true)))

@lru_cache(maxsize=10)
def cargar_modelo_y_scaler(id_item):
    modelo_path = f'modelo_{id_item}.keras'
    scaler_path = f'scaler_{id_item}.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    modelo = load_model(modelo_path, custom_objects={'root_mean_squared_error': root_mean_squared_error})
    return modelo, scaler

def predecir(x, model, scaler):
    y_pred_s = model.predict(x, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_s)
    return y_pred.flatten()

def reinterpolar_datos_por_id(datos):
    datos_reinterpolados = pd.DataFrame()
    for id_item in datos['id_item'].unique():
        df_item = datos[datos['id_item'] == id_item].copy()
        df_item = df_item.asfreq(freq='D', fill_value=0)
        df_item['id_item'] = id_item
        datos_reinterpolados = pd.concat([datos_reinterpolados, df_item])
    return datos_reinterpolados.reset_index()

def generar_predicciones_futuras(df, id_item, input_length, num_predicciones):
    modelo, scaler = cargar_modelo_y_scaler(id_item)
    ultima_fecha = df['Fecha'].iloc[-1]
    print("df\n")
    print(df)
    print("ultima_fecha\n")
    print(ultima_fecha)
    print(df.dtypes)
    fechas_futuras = [ultima_fecha + timedelta(days=i) for i in range(1, num_predicciones + 1)]
    ultimo_segmento = df['cantidad_frac'][-input_length:].values.reshape((1, input_length, 1))
    predicciones_futuras = []

    for _ in range(num_predicciones):
        prediccion = predecir(ultimo_segmento, modelo, scaler)
        prediccion_redondeada = round(prediccion[0])
        predicciones_futuras.append(prediccion_redondeada)
        ultimo_segmento = np.append(ultimo_segmento[:, 1:, :], np.array(prediccion_redondeada).reshape(1, 1, 1), axis=1)

    return pd.DataFrame({
        'Fecha': fechas_futuras,
        'Predicción': predicciones_futuras,
        'id_item': id_item
    })


def predecir_para_todos_los_items(datos, input_length, num_predicciones):
    resultados_totales = pd.DataFrame()
    lista = [90765, 27112, 13887, 79680, 1669, 101609, 54122, 88275, 13480, 39555]

    for id_item in lista:
        if id_item in datos['id_item'].values:
            # Filtrar los datos para el id_item actual
            df_item = datos[datos['id_item'] == id_item]
            # Generar predicciones para este id_item
            resultados_item = generar_predicciones_futuras(df_item, id_item, input_length, num_predicciones)
            # Concatenar los resultados al DataFrame total
            resultados_totales = pd.concat([resultados_totales, resultados_item])

    return resultados_totales

@app.route('/process_csv', methods=['POST'])
def process_csv():
    try:
        data = request.get_json()
        bucket_name = data['bucket']
        file_key = data['file_key']
        session_id = data['session_id']

        s3 = boto3.client('s3')
        local_file_path = '/tmp/' + file_key.split('/')[-1]
        s3.download_file(bucket_name, file_key, local_file_path)

        datos = leer_datos(local_file_path)
        print("datos\n")
        print(datos)
        print(datos.dtypes)
        datos = reinterpolar_datos_por_id(datos.groupby(['Fecha', 'id_item'], as_index=False).sum().set_index('Fecha'))
        resultados_futuros = predecir_para_todos_los_items(datos, 24, 4)
        resultados_futuros['Fecha'] = resultados_futuros['Fecha'].dt.strftime('%Y-%m-%d')

        # Notificación a Streamlit
        results_store[session_id] = resultados_futuros.to_dict(orient='records')  # Almacena los resultados

        return jsonify({"message": "Procesamiento completado en Flask", "session_id": session_id}), 200
    except Exception as e:
        print(f"Error procesando CSV: {str(e)}")
        return jsonify({"message": "Error procesando CSV", "error": str(e)}), 500


@app.route('/get_results/<session_id>', methods=['GET'])
def get_results(session_id):
    if session_id in results_store:
        return jsonify(results_store[session_id]), 200
    else:
        return jsonify({"error": "No hay resultados disponibles para este session_id"}), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
