import joblib
import streamlit as st
import os
import gzip
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Usamos una ruta relativa para acceder al directorio de 'models'
RUTA_MODELO = "/workspaces/fraud_para_Render/models/modelo_RandomForest_optimizado.pkl.gz"

# Función para cargar el modelo comprimido
def cargar_modelo_comprimido(ruta):
    """Carga el modelo comprimido con gzip."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"El archivo {ruta} no existe. Verifica que está en la carpeta models en Render.")

    with gzip.open(ruta, "rb") as f:
        modelo = joblib.load(f)
    print("✅ Modelo cargado exitosamente.")
    return modelo

# Cargar el modelo
try:
    model = cargar_modelo_comprimido(RUTA_MODELO)
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# Verificar si el modelo es correcto
if not isinstance(model, RandomForestClassifier):
    st.error("El archivo cargado no es un modelo RandomForest.")
    st.stop()

# Diccionario de clases
class_dict = {
    "0": "No Fraude",
    "1": "Fraude",
}

# Título de la aplicación
st.title("MODELO MACHINE LEARNING PARA LA PREDICCIÓN DE FRAUDE FINANCIERO")

# Menú de navegación
menu = st.sidebar.selectbox("Selecciona una opción", ["Predicción de Fraude", "Reseña sobre Fraudes Financieros"])

if menu == "Predicción de Fraude":
    st.header("Predicción de Fraude en Transacciones Bancarias")

    with st.form("Formulario de Datos"):
        st.subheader("Por favor, complete la siguiente información")
        
        income = st.number_input("Ingresos", min_value=0.0, max_value=1000000.0, step=1000.0)
        name_email_similarity = st.slider("Similitud entre Nombre y Email", min_value=0.0, max_value=1.0, step=0.01)
        prev_address_months_count = st.number_input("Meses en la Dirección Anterior", min_value=0, max_value=240, step=1)
        current_address_months_count = st.number_input("Meses en la Dirección Actual", min_value=0, max_value=240, step=1)
        customer_age = st.number_input("Edad del Cliente", min_value=18, max_value=100, step=1)
        intended_balcon_amount = st.number_input("Monto del Saldo Intencionado", min_value=0.0, max_value=1000000.0, step=1000.0)
        velocity_6h = st.number_input("Velocidad de Transacción en 6 Horas", min_value=0.0, max_value=1000.0, step=1.0)
        velocity_24h = st.number_input("Velocidad de Transacción en 24 Horas", min_value=0.0, max_value=1000.0, step=1.0)
        bank_branch_count_8w = st.number_input("Número de Sucursales Bancarias en 8 Semanas", min_value=0, max_value=20, step=1)
        date_of_birth_distinct_emails_4w = st.number_input("Correos Electrónicos Distintos en 4 Semanas", min_value=0, max_value=10, step=1)
        credit_risk_score = st.number_input("Puntuación de Riesgo Crediticio", min_value=0, max_value=1000, step=1)
        email_is_free = st.selectbox("Email es Gratuito", [0, 1])
        phone_home_valid = st.selectbox("Teléfono de Casa Válido", [0, 1])
        phone_mobile_valid = st.selectbox("Teléfono Móvil Válido", [0, 1])
        has_other_cards = st.selectbox("Tiene Otras Tarjetas", [0, 1])
        proposed_credit_limit = st.number_input("Límite de Crédito Propuesto", min_value=0.0, max_value=1000000.0, step=1000.0)
        foreign_request = st.selectbox("Solicitud Extranjera", [0, 1])
        keep_alive_session = st.number_input("Duración de la Sesión Activa", min_value=0.0, max_value=1440.0, step=1.0)
        device_distinct_emails_8w = st.number_input("Emails Distintos en 8 Semanas", min_value=0, max_value=10, step=1)
        month = st.slider("Mes", min_value=1, max_value=12, step=1)

        submit_button = st.form_submit_button(label="Predecir")

    if submit_button:
        # Lista de características en el mismo orden en que se entrenó el modelo
        columnas_correctas = [
            'income', 'name_email_similarity', 'prev_address_months_count',
            'current_address_months_count', 'customer_age', 'intended_balcon_amount',
            'velocity_6h', 'velocity_24h', 'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'email_is_free',
            'phone_home_valid', 'phone_mobile_valid', 'has_other_cards',
            'proposed_credit_limit', 'foreign_request', 'keep_alive_session',
            'device_distinct_emails_8w', 'month'
        ]

        # Crear DataFrame asegurando que las columnas coincidan
        data_df = pd.DataFrame([[
            income, name_email_similarity, prev_address_months_count, current_address_months_count, customer_age,
            intended_balcon_amount, velocity_6h, velocity_24h, bank_branch_count_8w,
            date_of_birth_distinct_emails_4w, credit_risk_score, email_is_free, phone_home_valid,
            phone_mobile_valid, has_other_cards, proposed_credit_limit, foreign_request,
            keep_alive_session, device_distinct_emails_8w, month
        ]], columns=columnas_correctas)

        # Realizar la predicción
        prediction = str(model.predict(data_df)[0])
        pred_class = class_dict[prediction]
        st.write("🔮 **Predicción:**", pred_class)

elif menu == "Reseña sobre Fraudes Financieros":
    st.header("Reseña sobre Fraudes Financieros")
    st.write("Explicación sobre los fraudes financieros...")
