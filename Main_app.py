import streamlit as st
import numpy as np
import pandas as pd
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
st.markdown("Explora conceptos de Procesamiento de Lenguaje Natural (NLP), Análisis de Tokens, Embeddings y Generación de Texto con la API de Groq.")

# ---------------------------------------------------------
# Sidebar: Configuración de la API Key y Parámetros del Modelo
# ---------------------------------------------------------
st.sidebar.header("🔑 Configuración de Groq")
api_key = st.sidebar.text_input("Ingresa tu API Key de Groq:", type="password")

# Modelos basados en la arquitectura Llama / Mistral / Gemma disponibles en Groq
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
    st.sidebar.warning("Por favor ingresa tu API Key de Groq para habilitar la generación de texto.")

# ---------------------------------------------------------
# Pestañas Principales
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔤 Tokens & Token IDs",
    "👜 Bag of Words (BoW)",
    "📐 Similitud y Embeddings",
    "🤖 Generación de Texto con Groq"
])

# ---------------------------------------------------------
# TAB 1: Tokens y Token IDs
# ---------------------------------------------------------
with tab1:
    st.header("Análisis de Tokens y IDs")
    st.markdown("Visualiza cómo el texto se divide en palabras/subpalabras (tokens) y sus identificadores numéricos simulados.")
    
    text_input_tokens = st.text_area("Ingresa un texto para tokenizar:", "La inteligencia artificial está transformando el desarrollo de software.", key="tokens_text")
    
    if text_input_tokens:
        # Tokenización basada en palabras / espacios (Simulación educacional de subpalabras)
        tokens = text_input_tokens.split()
        token_ids = [hash(token) % 50000 for token in tokens]  # Mapeo id simulado
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tokens")
            st.write(tokens)
        with col2:
            st.subheader("Token IDs (Simulados)")
            st.write(token_ids)
            
        df_tokens = pd.DataFrame({"Token": tokens, "Token ID": token_ids})
        st.dataframe(df_tokens.T)

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
        st.dataframe(df_bow)

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
        st.dataframe(df_sim)

# ---------------------------------------------------------
# TAB 4: Generación de Texto con LLM de Groq
# ---------------------------------------------------------
with tab4:
    st.header("Generación de Texto con Groq LLMs")
    
    prompt = st.text_area("Escribe tu Prompt o instrucción:", "Explica en tres viñetas qué es la similitud coseno y para qué sirve en NLP.")
    
    if st.button("Generar Respuesta"):
        if not client:
            st.error("Debes ingresar tu API Key de Groq en la barra lateral para generar texto.")
        else:
            try:
                with st.spinner("Procesando consulta con Groq..."):
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        model=model_option,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    
                    response_text = chat_completion.choices[0].message.content
                    st.subheader("Respuesta del Modelo:")
                    st.write(response_text)
            except Exception as e:
                st.error(f"Ocurrió un error al generar el texto: {e}")
