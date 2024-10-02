import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from datetime import datetime, timedelta
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_echarts import st_echarts
import folium
from streamlit_folium import st_folium
import time
import plotly.express as px
import altair as alt
import boto3
import websocket
from flask import Flask, request, jsonify
import requests
import uuid
import os
import pydeck as pdk

st.set_page_config(
    page_title="DASHBOARD",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.tu-ayuda.com',
        'Report a bug': 'https://www.tu-reporte.com',
        'About': '## Aplicación de Pronóstico de Ventas\nSe podrá visualizar el pronóstico de 10 productos junto a sus abastecimientos'
    }
)

# Decorador para cachear la función de carga de datos
@st.cache_data
def cargar_datos_csv(uploaded_file):
    return pd.read_csv(uploaded_file, delimiter=';')

@st.cache_data
def cargar_datos_geojson():
    return gpd.read_file('provinces.geojson')

def make_donut(input_response, input_text, input_color, color_theme='blues'):
    # Definir colores según el tema seleccionado
    chart_color = ['#29b5e8', '#155F7A'] if input_color == 'blue' else ['#27AE60', '#12783D']

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })

    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta=alt.Theta("% value:Q"),
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None)
    ).properties(width=130, height=130)

    text = plot.mark_text(align='center', color=chart_color[0], font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(
        text=alt.value(f'{input_response:.1f} %')
    )

    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta=alt.Theta("% value:Q"),
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None)
    ).properties(width=130, height=130)

    return plot_bg + plot + text

# Función para calcular totales por semana
def calcular_totales_semanales(df, columna_valor):
    df['Semana'] = df['Fecha'].dt.isocalendar().week
    semana_actual = df['Semana'].max()
    semana_anterior = semana_actual - 1

    total_semana_actual = df[df['Semana'] == semana_actual][columna_valor].sum()
    total_semana_anterior = df[df['Semana'] == semana_anterior][columna_valor].sum()

    delta = total_semana_actual - total_semana_anterior

    return total_semana_actual, delta


color_theme_dict = {
    'blues': ['#1f77b4', '#6baed6', '#9ecae1', '#c6dbef', '#e7f0fa', '#08306b', '#2171b5', '#4292c6', '#6baed6', '#9ecae1'],
    'cividis': ['#00204c', '#2e4e7e', '#5c75a1', '#869ac3', '#b4bfe2', '#e4e8f7', '#003f5c', '#58508d', '#bc5090', '#ff6361'],
    'greens': ['#006d2c', '#2ca02c', '#31a354', '#74c476', '#a1d99b', '#c7e9c0', '#00441b', '#006d2c', '#31a354', '#74c476'],
    'inferno': ['#000004', '#420a68', '#932667', '#dd513a', '#f99f27', '#fcffa4', '#b30000', '#e34a33', '#fdbb84', '#fee8c8'],
    'magma': ['#000004', '#3b0f6f', '#7d3c6b', '#ba5965', '#eb8060', '#febf87', '#fcfdbf', '#2c105c', '#711f81', '#b63679'],
    'plasma': ['#0d0887', '#4b03a1', '#7d03a8', '#a62196', '#d7437d', '#f47d5e', '#fdad4a', '#fdca33', '#f0f921', '#fffebe'],
    'reds': ['#b30000', '#e34a33', '#fc8d59', '#fdbb84', '#fdd49e', '#fee8c8', '#67000d', '#a50f15', '#cb181d', '#ef3b2c'],
    'rainbow': ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe'],
    'turbo': ['#30123b', '#4c1c64', '#6d308d', '#9b3fb0', '#d851da', '#fa7e84', '#fdcd56', '#fee32e', '#f5ff4f', '#fcfdbf'],
    'viridis': ['#440154', '#46327e', '#365c8d', '#277f8e', '#1fa187', '#4ac16d', '#a0da39', '#fde725', '#5ec962', '#35b779']
}



def mostrar_ventas_futuras_todos_items(datos, color_theme):
    # Verificar que la columna 'Fecha' sea de tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(datos['Fecha']):
        datos['Fecha'] = pd.to_datetime(datos['Fecha'])

    # Añadir una columna para el día de la semana en español
    dias_semana_es = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    datos['Día'] = datos['Fecha'].dt.day_name().map(dias_semana_es)

    # Crear una lista para almacenar las series de datos para cada item
    series = []

    for id_item in datos['id_item'].unique():
        # Filtrar los datos para el id_item específico
        datos_item = datos[datos['id_item'] == id_item]

        # Ordenar los datos según la fecha para mantener coherencia visual
        datos_item = datos_item.sort_values(by='Fecha')

        # Añadir los datos del item a la lista de series
        series.append({
            "name": f"Item {id_item}",
            "data": datos_item['Predicción'].tolist(),  # Predicciones diarias
            "type": "line",
            "smooth": True,
            "lineStyle": {"width": 2},
            "areaStyle": {"opacity": 0.2},
        })

    # Obtener los colores del tema seleccionado
    color_palette = color_theme_dict.get(color_theme, color_theme_dict['blues'])

    # Configurar las opciones para el gráfico
    options = {
        "title": {
            "text": "Predicción de Ventas por Día",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis"
        },
        "legend": {
            "data": [f"Item {id_item}" for id_item in datos['id_item'].unique()],
            "top": "bottom"
        },
        "xAxis": {
            "type": "category",
            "data": datos_item['Día'].tolist(),  # Usar los días presentes en los datos
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {"type": "value", "name": "Ventas Predichas"},
        "series": series,
        "color": color_palette  # Aplicar la paleta de colores
    }

    # Renderizar el gráfico con st_echarts
    st_echarts(options=options, height="500px")


def pydeck_ecuador_barra(datos2, col, color_theme):
    # Definir el diccionario de colores en formato RGB
    color_theme_dict = {
        'blues': [0, 128, 255],
        'cividis': [0, 64, 128],
        'greens': [34, 139, 34],
        'inferno': [255, 69, 0],
        'magma': [207, 70, 135],
        'plasma': [138, 43, 226],
        'reds': [220, 20, 60],
        'rainbow': [255, 0, 255],
        'turbo': [255, 165, 0],
        'viridis': [72, 61, 139]
    }

    # Obtener el color correspondiente al tema seleccionado
    fill_color = color_theme_dict.get(color_theme, [0, 128, 255])  # Default to 'blues' if theme not found

    # Crear el DataFrame para el mapa utilizando las columnas de latitud y longitud
    map_data = datos2[['latitud', 'longitud', 'cantidad_frac']].copy()
    map_data = map_data.rename(columns={'latitud': 'lat', 'longitud': 'lon', 'cantidad_frac': 'elevation'})

    # Filtrar valores inválidos (#N/D) y convertir a float
    map_data = map_data.replace('#N/D', float('nan'))
    map_data = map_data.dropna(subset=['lat', 'lon'])  # Eliminar filas con valores NaN
    map_data['lat'] = map_data['lat'].astype(float)
    map_data['lon'] = map_data['lon'].astype(float)

    # Configuración del mapa centrado en Ecuador
    view_state = pdk.ViewState(latitude=-1.831239, longitude=-78.183406, zoom=6, bearing=0, pitch=45)

    # Capa de columnas 3D
    column_layer = pdk.Layer(
        "ColumnLayer",
        map_data,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        radius=10000,  # Puedes ajustar el radio según la densidad de datos
        elevation_scale=100,  # Ajusta la escala de elevación para mejor visualización
        get_fill_color=fill_color,  # Aplicar el color del tema seleccionado
    )

    # Crear el mapa y mostrarlo en Streamlit
    r = pdk.Deck(layers=[column_layer], initial_view_state=view_state)
    col.pydeck_chart(r)



def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    input_df['Anio'] = pd.to_datetime(input_df['Fecha']).dt.year
    heatmap = alt.Chart(input_df).mark_rect().encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Year", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color(f'max({input_color}):Q',
                        legend=None,
                        scale=alt.Scale(scheme=input_color_theme)),  # Uso de input_color_theme
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(width=900).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    )
    return heatmap



def mostrar_top_10_provincias(datos, col, color_theme):
    # Definir el diccionario de colores
    color_theme_dict = {
        'blues': '#1f77b4',
        'cividis': '#00204c',
        'greens': '#2ca02c',
        'inferno': '#dd513a',
        'magma': '#ba5965',
        'plasma': '#d7437d',
        'reds': '#d62728',
        'rainbow': '#e6194b',
        'turbo': '#d851da',
        'viridis': '#35b779'
    }

    # Obtener el color correspondiente al tema seleccionado
    bar_color = color_theme_dict.get(color_theme, '#1f77b4')  # Default to 'blues' if theme not found

    # Agrupar por provincia y sumar la cantidad_frac
    top_provincias = datos.groupby('provincia')['cantidad_frac'].sum().reset_index()

    # Ordenar en orden descendente y seleccionar el top 10
    top_provincias_sorted = top_provincias.sort_values(by='cantidad_frac', ascending=False).head(10)

    # Mostrar el top 10 con barras de progreso simuladas
    col.markdown('#### Top 10 Provincias por Cantidad')

    for index, row in top_provincias_sorted.iterrows():
        provincia = row['provincia']
        cantidad = row['cantidad_frac']
        # Crear una barra de progreso utilizando Markdown y el color seleccionado
        bar = f"""
        <div style='background-color: {bar_color}; width: {cantidad/max(top_provincias_sorted.cantidad_frac)*100}%; height: 15px; margin-bottom: 5px;'></div>
        """
        col.markdown(f"**{provincia}**: {cantidad:.2f}")
        col.markdown(bar, unsafe_allow_html=True)

def validar_y_actualizar(fila):
    multiplicadores = {
        13887: 100,  # JERINGA MEGA INSUL 1MLx29Gx1/2x100
        54122: 10,   # XARELTO COM-RECx10MGx10
        88275: 28,
        39555 : 30
    }
    if fila["cantidad_unid"] >= 1:
        multiplicador = multiplicadores.get(fila['id_item'], 1)
        fila["cantidad_frac"] += (multiplicador * int(fila["cantidad_unid"]))
    return fila

def reinterpolar_datos_por_id(datos):
    datos_reinterpolados = pd.DataFrame()
    for id_item in datos['id_item'].unique():
        df_item = datos[datos['id_item'] == id_item].copy()
        df_item = df_item.asfreq(freq='D', fill_value=0)
        df_item['id_item'] = id_item
        datos_reinterpolados = pd.concat([datos_reinterpolados, df_item])
    return datos_reinterpolados.reset_index()


def mostrar_historico_y_predicciones(datos_ventas, datos_abastecimiento, datos_predicciones, color_theme):
    # Definir el diccionario de colores para los diferentes temas
    color_theme_dict = {
        'blues': ['#1f77b4', '#6baed6', '#9ecae1'],
        'cividis': ['#00204c', '#585b5d', '#a1a098'],
        'greens': ['#2ca02c', '#31a354', '#74c476'],
        'inferno': ['#dd513a', '#f99f27', '#fcffa4'],
        'magma': ['#ba5965', '#eb8060', '#febf87'],
        'plasma': ['#d7437d', '#f47d5e', '#fdca33'],
        'reds': ['#d62728', '#ff9896', '#9467bd'],
        'rainbow': ['#e6194b', '#3cb44b', '#ffe119'],
        'turbo': ['#d851da', '#fa7e84', '#fdcd56'],
        'viridis': ['#35b779', '#5ec962', '#fde725']
    }

    # Obtener la paleta de colores correspondiente al tema seleccionado
    color_palette = color_theme_dict.get(color_theme, ['#1f77b4', '#ff7f0e', '#2ca02c'])  # Default to 'blues' if theme not found

    # Preprocesar los datos
    datos_ventas = datos_ventas.drop(columns=["latitud", "longitud", "provincia", "id_item"]).groupby(['Fecha'], as_index=False).sum()
    datos_abastecimiento = datos_abastecimiento.drop(columns=["latitud", "longitud", "id_item", 'provincia']).groupby(['Fecha'], as_index=False).sum()
    datos_predicciones = datos_predicciones.drop(columns=["Día", "id_item"]).groupby(['Fecha'], as_index=False).sum()

    # Asegurarse de que las predicciones comiencen después de las ventas
    last_date_ventas = datos_ventas['Fecha'].max()
    datos_predicciones = datos_predicciones[datos_predicciones['Fecha'] > last_date_ventas]

    # Crear una lista de fechas para todo el período cubierto por ventas y predicciones
    all_dates = pd.date_range(start=datos_ventas['Fecha'].min(), end=datos_predicciones['Fecha'].max())

    # Crear una serie para las ventas (históricas y predicciones) y abastecimientos que cubra todas las fechas
    ventas_y_predicciones_series = pd.Series(index=all_dates)
    abastecimiento_series = pd.Series(index=all_dates)
    #predicciones_series = pd.Series(index=all_dates)

    # Rellenar las series con los datos correspondientes
    #ventas_series[datos_ventas['Fecha']] = datos_ventas['cantidad_frac'].values
    ventas_y_predicciones_series[datos_ventas['Fecha']] = datos_ventas['cantidad_frac'].values
    ventas_y_predicciones_series[datos_predicciones['Fecha']] = datos_predicciones['Predicción'].values
    abastecimiento_series[datos_abastecimiento['Fecha']] = datos_abastecimiento['cantidad_frac'].values
    #predicciones_series[datos_predicciones['Fecha']] = datos_predicciones['Predicción'].values

    # Convertir NaN a None para JSON (se convertirá a null)
    #series_ventas_data = ventas_series.replace({np.nan: None}).tolist()
    series_ventas_y_predicciones_data = ventas_y_predicciones_series.replace({np.nan: None}).tolist()
    series_abastecimiento_data = abastecimiento_series.replace({np.nan: None}).tolist()
    #series_predicciones_data = predicciones_series.replace({np.nan: None}).tolist()

    # Configurar las series para las líneas del gráfico
    series_ventas_y_predicciones  = {
        "name": "Ventas y Predicciones",
        "data": series_ventas_y_predicciones_data,
        "type": "line",
        "smooth": True,
        "lineStyle": {"width": 2},
        "areaStyle": {"opacity": 0.2},
    }

    series_abastecimiento = {
        "name": "Abastecimientos Históricos",
        "data": series_abastecimiento_data,
        "type": "line",
        "smooth": True,
        "lineStyle": {"width": 2, "type": "dashed"},
        "areaStyle": {"opacity": 0.2},
    }

    # Configurar las opciones para el gráfico
    options = {
        "title": {"text": "Histórico de Abastecimientos, Ventas y Predicciones Futuras", "left": "center"},
        "tooltip": {"trigger": "axis"},
        #"legend": {"data": ["Ventas Históricas", "Abastecimientos Históricos", "Predicciones Futuras"], "top": "bottom"},
        "legend": {"data": ["Ventas y Predicciones", "Abastecimientos Históricos"], "top": "bottom"},
        "xAxis": {"type": "category", "data": all_dates.strftime('%Y-%m-%d').tolist(), "axisLabel": {"rotate": 45}},
        "yAxis": {"type": "value", "name": "Cantidad"},
        "series": [
            {**series_ventas_y_predicciones, "lineStyle": {"color": color_palette[0]}},
            {**series_abastecimiento, "lineStyle": {"color": color_palette[1]}}
        ],
        "color": color_palette  # Aplicar la paleta de colores seleccionada
    }

    # Renderizar el gráfico con st_echarts
    st_echarts(options=options, height="500px")


def mostrar_historico_y_predicciones2(datos_ventas, datos_abastecimiento, datos_predicciones):
    datos_ventas = datos_ventas.drop(columns=["latitud", "longitud","provincia","id_item"])
    datos_abastecimiento = datos_abastecimiento.drop(columns=["latitud", "longitud","id_item",'provincia'])
    datos_predicciones = datos_predicciones.drop(columns=["Día","id_item"])


    datos_ventas = datos_ventas.groupby(['Fecha'], as_index=False).sum()
    datos_abastecimiento = datos_abastecimiento.groupby(['Fecha'], as_index=False).sum()
    datos_predicciones = datos_predicciones.groupby(['Fecha'], as_index=False).sum()


    # Verificar que las columnas 'Fecha' sean de tipo datetime
    # if not pd.api.types.is_datetime64_any_dtype(datos_ventas['Fecha']):
    datos_ventas['Fecha'] = pd.to_datetime(datos_ventas['Fecha'])
    # if not pd.api.types.is_datetime64_any_dtype(datos_abastecimiento['Fecha']):
    datos_abastecimiento['Fecha'] = pd.to_datetime(datos_abastecimiento['Fecha'])
    # if not pd.api.types.is_datetime64_any_dtype(datos_predicciones['Fecha']):
    datos_predicciones['Fecha'] = pd.to_datetime(datos_predicciones['Fecha'])


    # Asegurarse de que todos los datasets estén ordenados por fecha
    datos_ventas = datos_ventas.sort_values(by='Fecha')
    datos_abastecimiento = datos_abastecimiento.sort_values(by='Fecha')
    datos_predicciones = datos_predicciones.sort_values(by='Fecha')


    # Crear una serie para las ventas históricas
    series_ventas = {
        "name": "Ventas Históricas",
        "data": datos_ventas['cantidad_frac'].tolist(),
        "type": "line",
        "smooth": True,
        "lineStyle": {"width": 2},
        "areaStyle": {"opacity": 0.2},
    }

    # Crear una serie para los abastecimientos históricos
    series_abastecimiento = {
        "name": "Abastecimientos Históricos",
        "data": datos_abastecimiento['cantidad_frac'].tolist(),
        "type": "line",
        "smooth": True,
        "lineStyle": {"width": 2, "type": "dashed"},  # Línea discontinua para distinguir del histórico de ventas
        "areaStyle": {"opacity": 0.2},
    }

    # Crear una serie para las predicciones de ventas futuras
    series_predicciones = {
        "name": "Predicciones Futuras",
        "data": datos_predicciones['Predicción'].tolist(),
        "type": "line",
        "smooth": True,
        "lineStyle": {"width": 2, "color": "red"},  # Línea roja para destacar las predicciones
        "areaStyle": {"opacity": 0.2},
    }

    # print('Serie Prodicciones Futuras' + str(series_predicciones))

    # Configurar las opciones para el gráfico
    options = {
        "title": {
            "text": "Histórico de Abastecimientos, Ventas y Predicciones Futuras",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis"
        },
        "legend": {
            "data": ["Ventas Históricas", "Abastecimientos Históricos", "Predicciones Futuras"],
            "top": "bottom"
        },
        "xAxis": {
            "type": "category",
            "data": datos_ventas['Fecha'].dt.strftime('%Y-%m-%d').tolist(),  # Usar las fechas como etiquetas en el eje X
            "axisLabel": {"rotate": 45}
        },
        "yAxis": {"type": "value", "name": "Cantidad"},
        "series": [series_ventas, series_abastecimiento, series_predicciones],
        "color": ["#4E79A7", "#59A14F", "red"]
    }

    # Renderizar el gráfico con st_echarts
    st_echarts(options=options, height="500px")




def calcular_kpis_y_mostrar(datos_ventas, datos_abastecimiento, datos_predicciones):
    from streamlit_echarts import st_echarts

    # Verificar que las columnas 'Fecha' sean de tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(datos_ventas['Fecha']):
        datos_ventas['Fecha'] = pd.to_datetime(datos_ventas['Fecha'])
    if not pd.api.types.is_datetime64_any_dtype(datos_abastecimiento['Fecha']):
        datos_abastecimiento['Fecha'] = pd.to_datetime(datos_abastecimiento['Fecha'])
    if not pd.api.types.is_datetime64_any_dtype(datos_predicciones['Fecha']):
        datos_predicciones['Fecha'] = pd.to_datetime(datos_predicciones['Fecha'])

    # Filtrar datos para incluir solo el último mes
    fecha_hoy = pd.to_datetime("today")
    fecha_inicio = fecha_hoy - pd.DateOffset(days=30)

    datos_ventas_mes = datos_ventas[(datos_ventas['Fecha'] >= fecha_inicio) & (datos_ventas['Fecha'] <= fecha_hoy)]
    datos_abastecimiento_mes = datos_abastecimiento[(datos_abastecimiento['Fecha'] >= fecha_inicio) & (datos_abastecimiento['Fecha'] <= fecha_hoy)]

    # KPI 1: Tasa de Cobertura de Abastecimiento
    dias_cubiertos = (datos_abastecimiento_mes['cantidad_frac'] >= datos_ventas_mes['cantidad_frac']).sum()
    tasa_cobertura_abastecimiento = dias_cubiertos / len(datos_ventas_mes) * 100

    # KPI 2: Tasa de Crecimiento de Ventas
    ventas_mes_anterior = datos_ventas[(datos_ventas['Fecha'] >= fecha_inicio - pd.DateOffset(days=30)) &
                                       (datos_ventas['Fecha'] < fecha_inicio)]['cantidad_frac'].sum()
    ventas_mes_actual = datos_ventas_mes['cantidad_frac'].sum()
    tasa_crecimiento_ventas = ((ventas_mes_actual - ventas_mes_anterior) / ventas_mes_anterior) * 100

    # KPI 3: Promedio de Ventas Diarias
    promedio_ventas_diarias = datos_ventas_mes['cantidad_frac'].mean()

    # KPI 4: Promedio de Abastecimientos Diarios
    promedio_abastecimientos_diarios = datos_abastecimiento_mes['cantidad_frac'].mean()

    # KPI 5: Desviación Estándar de Ventas
    desviacion_ventas = datos_ventas_mes['cantidad_frac'].std()

    # KPI 6: Ratio de Ventas a Abastecimiento
    ratio_ventas_abastecimiento = ventas_mes_actual / datos_abastecimiento_mes['cantidad_frac'].sum()

    # Mostrar los KPIs
    st.write("### KPIs del Último Mes")
    st.metric("Tasa de Cobertura de Abastecimiento", f"{tasa_cobertura_abastecimiento:.2f}%")
    st.metric("Tasa de Crecimiento de Ventas", f"{tasa_crecimiento_ventas:.2f}%")
    st.metric("Promedio de Ventas Diarias", f"{promedio_ventas_diarias:.2f}")
    st.metric("Promedio de Abastecimientos Diarios", f"{promedio_abastecimientos_diarios:.2f}")
    st.metric("Desviación Estándar de Ventas", f"{desviacion_ventas:.2f}")
    st.metric("Ratio de Ventas a Abastecimiento", f"{ratio_ventas_abastecimiento:.2f}")

    # Aquí se puede incluir el gráfico previamente implementado
    mostrar_historico_y_predicciones(datos_ventas_mes, datos_abastecimiento_mes, datos_predicciones)


def upload_to_s3(file, bucket_name, object_name):
    # Crear una sesión de S3 usando boto3
    s3 = boto3.client('s3')
    try:
        # Subir el archivo
        s3.upload_fileobj(file, bucket_name, object_name)
        print("Archivo {} subido exitosamente a S3 en el bucket {}.".format(object_name, bucket_name))
    except Exception as e:
        st.error(f"Error subiendo el archivo: {e}")

def poll_for_results(session_id):
    flask_ip = os.getenv("FLASK_IP")
    while True:
        try:
            response = requests.get(f"http://50.19.133.115:80/get_results/{session_id}")
            print("response\n")
            print(response)
            if response.status_code == 200:
                return response.json()  # Retorna los resultados cuando estén listos
            else:
                print("Procesando en Flask, por favor espera...")
        except Exception as e:
            st.error(f"Error al obtener los resultados: {str(e)}")
            break
        time.sleep(1)


def preparar_datos1(datos):
    # Seleccionar columnas específicas
    columnas_especificas = ['Fecha','id_item', 'cantidad_unid', 'cantidad_frac','provincia', 'latitud', 'longitud']
    datos = datos[columnas_especificas]

    # Aplicar la función a cada fila
    datos = datos.apply(validar_y_actualizar, axis=1)
    datos = datos.drop(columns=["cantidad_unid"])

    # Convertir la columna Fecha a formato datetime
    datos['Fecha'] = pd.to_datetime(datos['Fecha'], format='%d/%m/%Y %H:%M')
    # Establecer la hora y el minuto a 0
    datos['Fecha'] = datos['Fecha'].apply(lambda dt: dt.replace(hour=0, minute=0, second=0))

    # Agrupar por Fecha e id_item y sumar cantidad_frac
    datos = datos.groupby(['Fecha', 'id_item', 'provincia', 'latitud', 'longitud'], as_index=False).sum()

    # Ordenar el dataset por Fecha
    datos.sort_index(inplace=False)
    #datos = datos.set_index('Fecha')

    return datos

# Preparar los datos
def preparar_datos(datos):
    # Reemplazar '#N/D' con NaN para unificar el manejo de valores faltantes
    datos.replace('#N/D', np.nan, inplace=True)

    # Seleccionar columnas específicas
    columnas_especificas = ['Fecha', 'id_item', 'cantidad_unid', 'cantidad_frac', 'provincia', 'latitud', 'longitud']
    datos = datos[columnas_especificas]

    # Aplicar la función a cada fila
    datos = datos.apply(validar_y_actualizar, axis=1)
    datos = datos.drop(columns=["cantidad_unid"])

    # Convertir la columna Fecha a formato datetime
    datos['Fecha'] = pd.to_datetime(datos['Fecha'], format='%d/%m/%Y %H:%M')
    # Establecer la hora y el minuto a 0
    datos['Fecha'] = datos['Fecha'].apply(lambda dt: dt.replace(hour=0, minute=0, second=0))

    # Reemplazar NaN en las columnas con valores predeterminados
    datos['provincia'].fillna('Desconocido', inplace=True)
    datos['latitud'].fillna(0, inplace=True)
    datos['longitud'].fillna(0, inplace=True)

    # Convertir columnas a string y reemplazar comas si necesario, luego convertir a float
    if datos['latitud'].dtype == 'object':
        datos['latitud'] = datos['latitud'].str.replace(',', '.').astype(float)
    else:
        datos['latitud'] = datos['latitud'].astype(float)

    if datos['longitud'].dtype == 'object':
        datos['longitud'] = datos['longitud'].str.replace(',', '.').astype(float)
    else:
        datos['longitud'] = datos['longitud'].astype(float)

    # Agrupar por Fecha, id_item, provincia, latitud y longitud, sumando cantidad_frac
    datos = datos.groupby(['Fecha', 'id_item', 'provincia', 'latitud', 'longitud'], as_index=False).sum()

    # Filtrar por una fecha y un id_item específicos
    fecha_deseada = pd.to_datetime('2024-07-12')
    result = datos[(datos['Fecha'] == fecha_deseada) & (datos['id_item'] == 88275)]

    # Ordenar el dataset por Fecha
    datos.sort_index(inplace=False)

    return datos

def validar_archivo(df, tipo_archivo):
    abast_columns = [
        "id_bodega_origen", "Codg_Orig", "Bodega_Origen", "tipo_cadena", "Codg_Dest",
        "Bodega_Dest", "Fecha", "id_item", "descripcion", "cantidad_unid",
        "cantidad_frac", "cantidad_solicitada", "costo_unitario_0", "costo_total_0",
        "id_Destino", "costo_total_0", "id_numero_doc", "descripcion", "observacion",
        "id_transferencia", "id_transfer_zeus", "estado_doc_egr", "estado_doc_ing",
        "latitud", "longitud", "provincia"
    ]
    ventas_columns = ['Fecha', 'Codigo', 'POS', 'id_item', 'Descripcion_larga', 'cantidad_unid', 'cantidad_frac', 'latitud', 'longitud', 'provincia']

    if tipo_archivo == 'abastecimiento':
        columnas_esperadas = abast_columns
    elif tipo_archivo == 'ventas':
        columnas_esperadas = ventas_columns
    else:
        raise ValueError("Tipo de archivo no reconocido")
    print("columnas esperadas\n")
    print(columnas_esperadas)
    print(df.columns)
    # Verifica si las columnas coinciden
    if set(df.columns) == set(columnas_esperadas):
        return True
    else:
        return False


def main():
    #st.title('Predicción de Ventas')
    st.markdown("<h1 style='text-align: center;'>DASHBOARD DE INVENTARIO</h1>", unsafe_allow_html=True)
    #col1, col2, col3 = st.columns([1.5, 4.5, 2], gap='medium')
    col1_1, col2_1 = st.columns([1,1], gap='medium')
    #datos = pd.DataFrame()

    uploaded_file = col1_1.file_uploader("Subir archivo de Ventas CSV", type="csv")
    uploaded_file2 = col2_1.file_uploader("Subir archivo de Abastecimientos CSV", type="csv")

    # Inicializar st.session_state.results si no existe
    if 'results' not in st.session_state:
        st.session_state.results = None

    if (uploaded_file and uploaded_file2) is not None:
        #datos = pd.read_csv(uploaded_file, delimiter=',')
        datos = cargar_datos_csv(uploaded_file)
        abast = cargar_datos_csv(uploaded_file2)
        print("datos columns\n")
        print(datos.columns)
        print("abast columns\n")
        print(abast.columns)
        ventas_valido = validar_archivo(datos, 'ventas')
        abast_valido = validar_archivo(abast, 'abastecimiento')
        if ventas_valido:
            # Barra de progreso
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)  # Simula un pequeño retardo
                progress_bar.progress(percent_complete + 1)

            st.success("Archivos subidos con éxito")

            print("uploaded_file: \n", uploaded_file.name)
            print("uploaded_file2: \n", uploaded_file2.name)

            with st.spinner('Esperando resultados...'): #st.container():
                start_time = time.time()  # Iniciar el cronómetro
                col1, col2, col3 = st.columns([1.5, 4.5, 2], gap='medium')
                # Cargar los datos
                datos2 = datos.copy()

                datos2 = preparar_datos(datos2)
                datos = preparar_datos1(datos)

                abast = preparar_datos(abast)

                datos = reinterpolar_datos_por_id(datos.groupby(['Fecha', 'id_item'], as_index=False).sum().set_index('Fecha'))
                abast = reinterpolar_datos_por_id(abast.groupby(['Fecha', 'id_item'], as_index=False).sum().set_index('Fecha'))

                print("DATOS2\n")
                print(datos2)

                # Obtener fechas mínima y máxima del dataset
                min_date = datos2['Fecha'].min().to_pydatetime()
                max_date = datos2['Fecha'].max().to_pydatetime()

                # Configuración de la barra lateral
                with st.sidebar:
                    ItemsId = datos['id_item'].unique()

                    # Filtrar opciones por defecto para que solo incluyan las que están en ItemsId
                    default_items = [item for item in [90765, 27112, 13887, 79680, 1669, 101609, 54122, 88275, 39555, 13480] if item in ItemsId]

                    lista = st.multiselect(
                        'Seleccionar ID del producto',
                        ItemsId,
                        default_items  # Solo los items que existen en ItemsId
                    )

                    # Crear un slider para el rango de fechas
                    selected_date = st.slider(
                        "Seleccionar rango de fecha",
                        min_value=min_date,
                        max_value=max_date,
                        value=(min_date, max_date),
                        step=timedelta(days=1),
                        format="DD/MM/YYYY"
                    )
                    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
                    input_color_theme = st.selectbox('Select a color theme', color_theme_list)

                # Filtrar los datos por el rango de fechas y los `id_item` seleccionados
                vent_filtrado = datos2[
                    (datos2['Fecha'] >= selected_date[0]) &
                    (datos2['Fecha'] <= selected_date[1]) &
                    (datos2['id_item'].isin(lista))

                ]

                # Filtrar los datos de abastecimiento por el rango de fechas y los `id_item`
                abast_filtrado = abast[
                    (abast['Fecha'] >= selected_date[0]) &
                    (abast['Fecha'] <= selected_date[1]) &
                    (abast['id_item'].isin(lista))
                ]

                #Filtrar los datos de predicciones por 'id_item' seleccionados
                datos_filtradosFut = datos[
                    (datos['id_item'].isin(lista))

                ]

                vent_filtrado_org = vent_filtrado.copy()
                abast_filtrado_org = abast_filtrado.copy()

                if vent_filtrado.empty:
                    col1.warning("No hay datos disponibles para los filtros seleccionados")
                else:

                    session_id = str(uuid.uuid4())
                    file_name = f"{session_id}_filtered.csv"
                    bucket_name = 'integradora-bucket'
                    datos_filtradosFut.to_csv(file_name, sep=';', index=False)
                    with open(file_name, 'rb') as f:
                        upload_to_s3(f, bucket_name, file_name)

                    os.remove(file_name)
                    # Enviar archivo a Flask para procesamiento
                    results = poll_for_results(session_id)
                    print("results\n")
                    print(results)

                    if results:
                        end_time = time.time()  # Detener el cronómetro al obtener los resultados
                        processing_time = end_time - start_time  # Calcular el tiempo total
                        print("Tiempo total de procesamiento: {:.2f} segundos".format(processing_time))
                        df = pd.DataFrame(results)
                        pydeck_ecuador_barra(vent_filtrado, col2, input_color_theme)
                        col2.write("Heatmap de Cantidad de Registros por Año y Provincia")
                        heatmap = make_heatmap(vent_filtrado, 'Anio', 'provincia', 'cantidad_frac', input_color_theme)
                        col2.altair_chart(heatmap, use_container_width=True)
                        mostrar_top_10_provincias(vent_filtrado, col3, input_color_theme)
                        col3.write("Resultados de Predicción:")
                        col3.dataframe(df)
                        with col2:
                            #calcular_kpis_y_mostrar(datos2, abast,resultados_futuros )
                            mostrar_ventas_futuras_todos_items(df, input_color_theme)


                            mostrar_historico_y_predicciones(vent_filtrado_org, abast_filtrado_org, df, input_color_theme)
                            total_ventas, delta_ventas = calcular_totales_semanales(vent_filtrado_org, 'cantidad_frac')
                            col1.metric(label="Total de Unidades Vendidas", value=total_ventas, delta=int(delta_ventas))

                            total_abast, delta_abast = calcular_totales_semanales(abast_filtrado_org, 'cantidad_frac')
                            col1.metric(label="Total de Unidades Abastecidas", value=total_abast, delta=int(delta_abast))

                            total_predicciones, delta_predicciones = calcular_totales_semanales(df, 'Predicción')
                            col1.metric(label="Total de Predicciones de Ventas", value=total_predicciones, delta=int(delta_predicciones))

                            promedio_ventas_diarias = total_ventas / vent_filtrado_org['Fecha'].nunique()
                            cobertura_inventario = total_abast / promedio_ventas_diarias if promedio_ventas_diarias > 0 else 0
                            col1.metric(label="Cobertura de Inventario (días)", value=f"{cobertura_inventario:.2f}")

                            st.header("Gráfico de Ventas y Predicciones Futuras")
                            ventas_para_grafico = vent_filtrado_org.groupby('Fecha')['cantidad_frac'].sum().reset_index().rename(columns={'cantidad_frac': 'Ventas'})
                            predicciones_para_grafico = df[['Fecha', 'Predicción']].rename(columns={'Predicción': 'Ventas'})
                            ventas_futuras_combinadas = pd.concat([ventas_para_grafico, predicciones_para_grafico], ignore_index=True).sort_values(by='Fecha')
                            st.line_chart(ventas_futuras_combinadas.set_index('Fecha'))

                            tasa_cumplimiento = max(0, min(100, (total_abast / total_predicciones) * 100)) if total_predicciones > 0 else 0
                            col1.write('Tasa de Cumplimiento de Abastecimiento')
                            donut_chart = make_donut(tasa_cumplimiento, "Cumplido", "green")
                            col1.altair_chart(donut_chart)

                            cobertura_inventario = max(0, min(100, (total_abast / promedio_ventas_diarias))) if promedio_ventas_diarias > 0 else 0
                            col1.write('Cobertura de Inventario')
                            donut_chart2 = make_donut(cobertura_inventario, "Cubierto", "blue")
                            col1.altair_chart(donut_chart2)

                            # Crear el expander
                            with st.expander('Ventas por día'):

                                # Agregar información del contacto del creador
                                st.markdown("**Contactos**")

                                # Cargar y mostrar ícono de GitHub
                                st.image("github.png", width=24)
                                st.markdown("[GitHub](https://github.com/JeanVillamar)")

                                # Cargar y mostrar ícono de LinkedIn
                                st.image("linkedin.png", width=24)
                                st.markdown("[LinkedIn](www.linkedin.com/in/jean-villamar)")
                                #mostrar_top_10_provincias(vent_filtrado, col3)

                    else:
                        st.error("No se pudieron obtener los resultados.")
        else:
            if not ventas_valido:
                st.error("El archivo de ventas no tiene el formato esperado.")
            if not abast_valido:
                st.error("El archivo de abastecimiento no tiene el formato esperado.")

if __name__ == '__main__':
    main()