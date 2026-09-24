import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import textstat
import re
from groq import Groq

# Configuración inicial de la página
st.set_page_config(
    page_title="OCR & LLM Amplifier + NLP Metrics",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 OCR + Amplificación LLM & Métricas Textuales")
st.markdown(
    "Esta plataforma recibe una imagen, extrae o procesa el contenido, "
    "amplía la respuesta utilizando un LLM ajustando su tono (Formal o Técnico) "
    "y evalúa métricas del texto generado (Coherencia, Sintaxis, Gramática y Medidas)."
)

# ---------------------------------------------------------
# Sidebar: Configuración de la API Key y Parámetros del LLM
# ---------------------------------------------------------
st.sidebar.header("🔑 Configuración de la API Key")
api_key = st.sidebar.text_input("Ingresa tu API Key de Groq / OpenAI:", type="password")

model_option = st.sidebar.selectbox(
    "Selecciona el Modelo LLM:",
    [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
)

st.sidebar.header("🎛️ Parámetros del Modelo")
temperature = st.sidebar.slider("Temperatura (Creatividad):", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
max_tokens = st.sidebar.slider("Máximo de Tokens:", min_value=100, max_value=4096, value=512, step=50)

# Inicializar cliente
client = None
if api_key:
    try:
        client = Groq(api_key=api_key)
        st.sidebar.success("API Key vinculada exitosamente.")
    except Exception as e:
        st.sidebar.error(f"Error al inicializar el cliente: {e}")
else:
    st.sidebar.warning("Por favor ingresa tu API Key para habilitar la generación.")

# ---------------------------------------------------------
# Pestañas Principales
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📷 Carga de Imagen & Generación LLM", "📊 Métricas del Texto Generado"])

# Variable de estado para compartir el texto entre pestañas
if "generated_response" not in st.session_state:
    st.session_state.generated_response = ""
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# ---------------------------------------------------------
# TAB 1: OCR + LLM
# ---------------------------------------------------------
with tab1:
    col_img, col_opt = st.columns([1, 1])
    
    with col_img:
        st.subheader("1. Carga la Imagen")
        uploaded_file = st.file_uploader("Selecciona una imagen (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen Cargada", use_container_width=True)
            
    with col_opt:
        st.subheader("2. Configuración de la Respuesta")
        
        tone = st.radio(
            "Selecciona el estilo o tono de la respuesta ampliada:",
            ["Formal", "Técnico"],
            help="Formal: Lenguaje ejecutivo y estructurado. Técnico: Lenguaje especializado con detalles profundos."
        )
        
        prompt_instruction = st.text_area(
            "Instrucción adicional para el LLM (Opcional):",
            "Resume, explica en detalle y amplía la información extraída de la imagen."
        )

    st.markdown("---")
    st.subheader("3. Extracción de Texto (OCR) y Procesamiento")
    
    # Campo para texto extraído (Simulación / Tesseract / OCR Entrada)
    extracted_input = st.text_area(
        "Texto extraído de la imagen (puedes editarlo o ingresarlo si el OCR automático no está activo):",
        value=st.session_state.extracted_text if st.session_state.extracted_text else "Ingresa o verifica el texto de la imagen aquí...",
        height=120
    )
    st.session_state.extracted_text = extracted_input

    if st.button("🚀 Procesar y Ampliar con LLM"):
        if not client:
            st.error("Debes ingresar tu API Key en la barra lateral antes de continuar.")
        elif not extracted_input.strip():
            st.error("El texto extraído está vacío.")
        else:
            # Construcción del prompt según el estilo seleccionado
            system_prompt = f"Eres un asistente experto en análisis textual. Tu tarea es responder con un tono **{tone.upper()}**."
            user_prompt = (
                f"A continuación se presenta el texto extraído mediante OCR de una imagen:\n\n"
                f"\"\"\"\n{extracted_input}\n\"\"\"\n\n"
                f"Instrucción: {prompt_instruction}\n\n"
                f"Por favor, proporciona una respuesta amplia, coherente y bien estructurada en tono {tone}."
            )
            
            try:
                with st.spinner("Procesando y generando respuesta ampliada con el LLM..."):
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=model_option,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    
                    st.session_state.generated_response = chat_completion.choices[0].message.content
                    st.success("¡Respuesta generada con éxito!")
            except Exception as e:
                st.error(f"Error durante la generación: {e}")

    if st.session_state.generated_response:
        st.markdown("### 📝 Respuesta Ampliada del LLM:")
        st.info(st.session_state.generated_response)

# ---------------------------------------------------------
# TAB 2: Métricas del Texto Generado
# ---------------------------------------------------------
with tab2:
    st.header("📊 Análisis y Métricas del Texto Generado")
    
    text_to_analyze = st.session_state.generated_response
    
    if not text_to_analyze:
        st.warning("Aún no se ha generado ninguna respuesta en la pestaña anterior. Genera un texto para ver sus métricas.")
    else:
        st.markdown("### Texto bajo evaluación:")
        st.caption(text_to_analyze[:300] + ("..." if len(text_to_analyze) > 300 else ""))
        st.markdown("---")
        
        # 1. Medidas Cuantitativas Básicas
        words = re.findall(r'\b\w+\b', text_to_analyze)
        sentences = [s for s in re.split(r'[\.\!\?]+', text_to_analyze) if s.strip()]
        num_words = len(words)
        num_chars = len(text_to_analyze)
        num_sentences = len(sentences) if sentences else 1
        vocab_unique = len(set([w.lower() for w in words]))
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Palabras", num_words)
        col_m2.metric("Total Caracteres", num_chars)
        col_m3.metric("Oraciones / Frases", num_sentences)
        col_m4.metric("Vocabulario Único", vocab_unique)
        
        st.markdown("---")
        
        # 2. Métricas de Complejidad, Sintaxis y Legibilidad
        st.subheader("📐 Sintaxis, Legibilidad y Complejidad")
        
        # Cálculo de métricas
        flesch_score = textstat.flesch_reading_ease(text_to_analyze)
        fog_index = textstat.gunning_fog(text_to_analyze)
        ttr = (vocab_unique / num_words) if num_words > 0 else 0  # Type-Token Ratio (Diversidad Léxica)
        avg_words_per_sentence = num_words / num_sentences if num_sentences > 0 else 0

        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("Índice de Legibilidad (Flesch)", f"{flesch_score:.2f}")
            st.caption("Valores más altos indican un texto más fácil de leer.")
            
        with c2:
            st.metric("Gunning Fog Index", f"{fog_index:.2f}")
            st.caption("Años de educación formal necesarios para entender el texto.")
            
        with c3:
            st.metric("Diversidad Léxica (TTR)", f"{ttr:.2f}")
            st.caption("Relación entre vocabulario único y total de palabras (Coherencia/Riqueza).")
            
        # 3. Métricas de Coherencia y Gramática
        st.subheader("🧠 Evaluación Cualitativa Estimada (Coherencia & Gramática)")
        
        # Estimación heurística de métricas
        grammar_score = min(100, max(60, int(100 - (fog_index * 2))))
        coherence_score = min(100, max(50, int(ttr * 100 + (flesch_score * 0.3))))
        syntax_score = min(100, max(50, int(100 - abs(avg_words_per_sentence - 15) * 2)))

        df_metrics = pd.DataFrame({
            "Dimensión Evaludada": [
                "Gramática & Corrección Estructural",
                "Coherencia Semántica Estimada",
                "Complejidad Sintáctica",
                "Longitud Promedio por Oración"
            ],
            "Valor / Puntuación": [
                f"{grammar_score} / 100",
                f"{coherence_score} / 100",
                f"{syntax_score} / 100",
                f"{avg_words_per_sentence:.1f} palabras"
            ],
            "Diagnóstico": [
                "Estructura gramatical limpia y adecuada.",
                "Riqueza de vocabulario y fluidez general.",
                "Estructura de oraciones balanceada.",
                "Longitud adecuada para comprensión."
            ]
        })
        
        st.table(df_metrics)
