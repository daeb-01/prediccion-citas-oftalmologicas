# Cargamos librerías principales
import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Cargamos el modelo
filename = 'clasificacion.pkl'
modelo, labelencoder, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

# Interfaz gráfica
st.title('Predicción de asistencia a citas oftalmológicas')

# Captura de datos
edad = st.slider('Edad', min_value=0, max_value=100, value=40, step=1)

nivel_socioeconomico = st.selectbox(
    'Nivel socioeconómico',
    [1, 2, 3, 4, 5, 6]
)

especialidad = st.selectbox(
    'Especialidad',
    [
        'BAJA_VISION',
        'CORNEA',
        'ESTRABISMO',
        'GLAUCOMA',
        'NEUROFTALMOLOGIA',
        'OCULOPLASTIA',
        'OFTALMOLOGIA',
        'OFTALMOPEDIATRIA',
        'OPTOMETRIA-OPTICA',
        'PROTESIS-OCULAR',
        'RETINA',
        'UVEITIS'
    ]
)

dia_semana = st.selectbox(
    'Día de la semana',
    ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado']
)

rango_horario = st.selectbox(
    'Rango horario',
    ['MANANA', 'MEDIO_DIA', 'TARDE']
)

dias_antelacion = st.number_input(
    'Días de antelación',
    min_value=0,
    max_value=365,
    value=30
)

tipo_cita = st.selectbox(
    'Tipo de cita',
    ['CONTROL', 'POST-OPERATORIA', 'PRIMERA_VEZ', 'REVISION']
)

genero = st.selectbox(
    'Género',
    ['FEMENINO', 'MASCULINO']
)

# DataFrame con los datos ingresados
datos = [[
    nivel_socioeconomico,
    especialidad,
    dia_semana,
    rango_horario,
    dias_antelacion,
    tipo_cita,
    edad,
    genero
]]

data = pd.DataFrame(
    datos,
    columns=[
        'NIVEL_SOCIOECONOMICO',
        'ESPECIALIDAD',
        'DIA_SEMANA',
        'RANGO_HORARIO',
        'DIAS_ANTELACION',
        'TIPO_CITA',
        'EDAD',
        'GENERO'
    ]
)

# Preparación de datos
data_preparada = data.copy()

multicat = ['ESPECIALIDAD', 'DIA_SEMANA', 'RANGO_HORARIO', 'TIPO_CITA']

data_preparada = pd.get_dummies(
    data_preparada,
    columns=multicat,
    drop_first=False,
    dtype=int
)

data_preparada = pd.get_dummies(
    data_preparada,
    columns=['GENERO'],
    drop_first=True,
    dtype=int
)

# Ajustar columnas al modelo
data_preparada = data_preparada.reindex(
    columns=variables,
    fill_value=0
)

# Normalización
variables_numericas = [
    'NIVEL_SOCIOECONOMICO',
    'DIAS_ANTELACION',
    'EDAD'
]

data_preparada[variables_numericas] = min_max_scaler.transform(
    data_preparada[variables_numericas]
)

# Predicción
prediccion = modelo.predict(data_preparada)
prediccion = labelencoder.inverse_transform(prediccion)

# Mostrar resultado
st.subheader('Resultado de la predicción')

if prediccion[0] == 'CUMPLIDA':
    st.success('✅ CUMPLIDA')
else:
    st.warning('⚠️ NO_CUMPLIDA')

# Mostrar datos utilizados
data['PREDICCION'] = prediccion
st.dataframe(data)