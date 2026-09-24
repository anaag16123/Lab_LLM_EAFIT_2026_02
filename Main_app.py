import streamlit as st
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Configuración de la página
st.set_page_config(
    page_title="Streamlit LLM & NLP Playground",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Streamlit LLM & Groq NLP Playground")
st.markdown("Explora conceptos de Tokenización con colores, Bag of Words, Similitud, Embeddings y Comparación de Temperaturas con la API de Groq.")

# ---------------------------------------------------------
# Sidebar: Configuración de la API Key y Parámetros
# ---------------------------------------------------------
st.sidebar.header("🔑 Configuración de Groq")
api_key = st.sidebar.text_input("Ingresa tu API Key de Groq:", type="password")

model_option = st.sidebar.selectbox(
    "Selecciona el modelo de Groq:",
    [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
)

temperature = st.sidebar.slider("Temperatura (Creatividad):", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider("Máximo de Tokens a Generar:", min_value=10, max_value=2048, value=256, step=10)

# Inicializar cliente de Groq
client = None
if api_key:
    try:
        client = Groq(api_key=api_key)
        st.sidebar.success("API Key cargada correctamente.")
    except Exception as e:
        st.sidebar.error(f"Error al conectar con Groq: {e}")
else:
    st.sidebar.warning("Ingresa tu API Key de Groq para habilitar las funciones de generación de texto.")

# ---------------------------------------------------------
# Pestañas Principales
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎨 Tokenización Visual & IDs",
    "👜 Bag of Words (BoW)",
    "📐 Similitud y Embeddings",
    "🤖 Generación de Texto y Comparador"
])

# ---------------------------------------------------------
# TAB 1: Tokens, Token IDs y Visualización con Colores
# ---------------------------------------------------------
with tab1:
    st.header("Análisis y Visualización de Tokens")
    st.markdown("Visualiza la partición de texto con **diferentes métodos** y observa las fichas (tokens) resaltadas con distintos colores.")
    
    text_input_tokens = st.text_area(
        "Ingresa un texto para tokenizar:",
        "La inteligencia artificial está transformando la manera en que procesamos el lenguaje humano.",
        key="tokens_text"
    )
    
    method = st.selectbox(
        "Selecciona el método de tokenización:",
        ["Por Palabras (Palabra completa)", "Por Caracteres", "Por Subpalabras (N-grams de 3-4 caracteres)"]
    )
    
    if text_input_tokens:
        # Aplicar método seleccionado
        if method == "Por Palabras (Palabra completa)":
            tokens = text_input_tokens.split()
        elif method == "Por Caracteres":
            tokens = list(text_input_tokens)
        else: # Subpalabras / N-grams
            tokens = [text_input_tokens[i:i+4] for i in range(0, len(text_input_tokens), 4)]
        
        # Generar Token IDs (Mapeo determinista educacional)
        token_ids = [abs(hash(t)) % 50000 for t in tokens]
        
        # Paleta de colores pastel en CSS
        colors = [
            "#FFD1DC", "#B19FFB", "#C1E1C1", "#FFDFBA", "#FFFFBA",
            "#BAE1FF", "#E8AEB7", "#B28DFF", "#D5AAFF", "#AFF8D8"
        ]
        
        st.subheader("🎨 Visualización de Particiones")
        
        # Generar HTML coloreado
        html_content = "<div style='line-height: 2.2; font-size: 18px; font-family: monospace; padding: 15px; border-radius: 8px; background-color: #f8f9fa;'>"
        for idx, (token, t_id) in enumerate(zip(tokens, token_ids)):
            color = colors[idx % len(colors)]
            display_token = token.replace(" ", "␣") if token == " " else token
            html_content += f"<span style='background-color: {color}; padding: 3px 7px; margin: 2px; border-radius: 5px; border: 1px solid #ccc; font-weight: bold; color: #111;' title='ID: {t_id}'>{display_token}</span>"
        html_content += "</div>"
        
        st.markdown(html_content, unsafe_allow_html=True)
        st.caption("Pasa el cursor sobre cada token resaltado para ver su Token ID simulado.")
        
        # Tabla detallada
        st.subheader("Tabla de Tokens e IDs")
        df_tokens = pd.DataFrame({"Token": tokens, "Token ID": token_ids})
        st.dataframe(df_tokens, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Bag of Words (BoW)
# ---------------------------------------------------------
with tab2:
    st.header("Modelo Bag of Words (Bolsa de Palabras)")
    st.markdown("Representa el texto contando la frecuencia de aparición de cada palabra en un conjunto de documentos.")
    
    doc1 = st.text_input("Documento 1:", "Me gusta la inteligencia artificial y el aprendizaje automático.")
    doc2 = st.text_input("Documento 2:", "El aprendizaje automático es una rama de la inteligencia artificial.")
    
    if doc1 and doc2:
        corpus = [doc1, doc2]
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(corpus)
        
        df_bow = pd.DataFrame(
            bow_matrix.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=["Documento 1", "Documento 2"]
        )
        
        st.subheader("Matriz Frecuencia de Términos (BoW)")
        st.dataframe(df_bow, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Métricas de Similitud y Embeddings
# ---------------------------------------------------------
with tab3:
    st.header("Métricas de Similitud y Vectores (Embeddings)")
    st.markdown("Compara qué tan similares son dos frases utilizando la **Similitud Coseno** sobre vectores de frecuencia/características.")
    
    s1 = st.text_input("Frase A:", "Me encanta programar aplicaciones en Python.")
    s2 = st.text_input("Frase B:", "Desarrollar código en Python es genial.")
    
    if s1 and s2:
        vec = CountVectorizer()
        matrix = vec.fit_transform([s1, s2])
        sim_score = cosine_similarity(matrix[0], matrix[1])[0][0]
        
        st.metric(label="Similitud Coseno (0 a 1)", value=f"{sim_score:.4f}")
        
        st.subheader("Vectores de Características (Frecuencia)")
        df_sim = pd.DataFrame(
            matrix.toarray(),
            columns=vec.get_feature_names_out(),
            index=["Frase A", "Frase B"]
        )
        st.dataframe(df_sim, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: Generación de Texto y Comparador de Temperatura
# ---------------------------------------------------------
with tab4:
    st.header("Generación de Texto & Comparativa de Parámetros")
    
    prompt = st.text_area(
        "Escribe tu Prompt o instrucción:",
        "Escribe un poema corto de 4 versos sobre la creatividad de la Inteligencia Artificial."
    )
    
    st.markdown("---")
    
    # Modo 1: Generación Individual
    st.subheader("1. Generación Individual con Parámetros del Sidebar")
    if st.button("Generar Respuesta Individual"):
        if not client:
            st.error("Por favor, ingresa tu API Key de Groq en la barra lateral.")
        else:
            try:
                with st.spinner("Generando respuesta..."):
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_option,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    st.success("Respuesta Generada:")
                    st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Error al generar texto: {e}")

    st.markdown("---")
    
    # Modo 2: Comparación de Temperaturas
    st.subheader("2. Comparador de Respuestas por Temperatura")
    st.markdown("Genera dos respuestas simultáneas variando la **Temperatura** para observar la diferencia en variabilidad y creatividad.")
    
    col_temp1, col_temp2 = st.columns(2)
    with col_temp1:
        temp1 = st.slider("Temperatura A (Baja - Más Preciso/Determinado):", 0.0, 2.0, 0.2, 0.1)
    with col_temp2:
        temp2 = st.slider("Temperatura B (Alta - Más Creativo/Variado):", 0.0, 2.0, 1.2, 0.1)
        
    if st.button("Comparar Respuestas"):
        if not client:
            st.error("Por favor, ingresa tu API Key de Groq en la barra lateral.")
        else:
            try:
                with st.spinner("Generando ambas respuestas con Groq..."):
                    # Petición 1
                    res1 = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_option,
                        temperature=temp1,
                        max_tokens=max_tokens,
                    )
                    # Petición 2
                    res2 = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_option,
                        temperature=temp2,
                        max_tokens=max_tokens,
                    )
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.info(f"**Resultado A (Temperatura = {temp1})**")
                        st.write(res1.choices[0].message.content)
                    with c2:
                        st.info(f"**Resultado B (Temperatura = {temp2})**")
                        st.write(res2.choices[0].message.content)
            except Exception as e:
                st.error(f"Error al comparar respuestas: {e}")
