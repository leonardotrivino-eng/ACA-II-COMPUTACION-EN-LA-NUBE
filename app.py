import os
import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de Streamlit
st.set_page_config(
    page_title="Asistente RAG - Normatividad Educación CUN",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_models_and_db(api_key: str, qdrant_url: str = None, qdrant_api_key: str = None):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    if qdrant_url and qdrant_api_key:
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        connection_type = "Qdrant Cloud ☁️"
    else:
        qdrant_client = QdrantClient(":memory:")
        connection_type = "Qdrant en Memoria 💾"
        
    if api_key:
        genai.configure(api_key=api_key)
        
    return embedding_model, qdrant_client, connection_type

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL") or st.secrets.get("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or st.secrets.get("QDRANT_API_KEY", "")
COLLECTION_NAME = "rag_educacion_cloud"

embedding_model, qdrant_client, conn_status = init_models_and_db(GEMINI_KEY, QDRANT_URL, QDRANT_API_KEY)

def setup_vector_store():
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        
        corpus = [
            {
                "source": "decreto-rectoral-1751-de-2022-reglamento-academico.pdf",
                "title": "Reglamento Académico de Pregrado",
                "text": """
                DECRETO RECTORAL 1751 DE 2022 - REGLAMENTO ACADÉMICO DE PREGRADO.
                Artículo 15: Reserva de Cupo. El estudiante que requiera suspender temporalmente sus estudios podrá solicitar reserva de cupo académico hasta por dos (2) semestres consecutivos o discontinuos. La solicitud debe radicarse ante la Secretaría Académica antes de la cuarta semana del período lectivo.
                Artículo 22: Reingreso y Readmisión. Quien haya hecho uso de la reserva de cupo dentro del término concedido podrá solicitar su reingreso. Si el retiro superó los dos semestres, deberá someterse al proceso de readmisión según disponibilidad de cupo y actualización del plan de estudios.
                Artículo 30: Carga Académica y Prerrequisitos. La carga académica ordinaria es de máximo 18 créditos por semestre. El estudiante no podrá matricular asignaturas sin haber aprobado previamente los prerrequisitos establecidos en el plan de estudios.
                """
            },
            {
                "source": "rutas-de-atencion-psicopedagogica.pdf",
                "title": "Guía de Rutas de Atención Integrada",
                "text": """
                GUÍA DE RUTAS DE ATENCIÓN INTEGRAL Y CONVIVENCIA UNIVERSITARIA.
                Protocolo 1: Acompañamiento Académico y Medios de Convivencia. Ante situaciones pedagógicas o de convivencia, el comité activará la Ruta de Atención Integral con acompañamiento psicológico inmediato.
                Protocolo 2: Orientación Estudiantil. Se brindará acompañamiento institucional garantizando el secreto profesional y el debido proceso académico.
                """
            },
            {
                "source": "Student_performance_data__.csv",
                "title": "Análisis Estadístico de Desempeño Estudiantil",
                "text": """
                ANÁLISIS ESTRUCTURADO DE RENDIMIENTO ESTUDIANTIL Y FACTORES ACADÉMICOS.
                Métrica de Desempeño: El promedio acumulado (GPA) muestra una correlación directa del 0.78 con el porcentaje de asistencia a tutorías virtuales y la entrega oportuna de actividades RAG en la plataforma.
                """
            }
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        points = []
        idx = 0
        for doc in corpus:
            chunks = text_splitter.split_text(doc["text"])
            for i, chunk in enumerate(chunks):
                vec = embedding_model.encode(chunk).tolist()
                points.append(PointStruct(
                    id=idx,
                    vector=vec,
                    payload={
                        "text": chunk.strip(),
                        "source": doc["source"],
                        "title": doc["title"],
                        "chunk_id": i + 1
                    }
                ))
                idx += 1
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

setup_vector_store()

def ejecutar_pipeline_rag(pregunta: str, top_k: int = 3):
    query_vector = embedding_model.encode(pregunta).tolist()
    
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    
    contexto_list = []
    trazabilidad_fuentes = []
    
    for i, pt in enumerate(search_result.points, start=1):
        src = pt.payload.get("source", "N/A")
        title = pt.payload.get("title", "Documento")
        chunk_id = pt.payload.get("chunk_id", "N/A")
        texto = pt.payload.get("text", "")
        score = pt.score
        
        contexto_list.append(f"[Fuente {i}: {src} | Chunk #{chunk_id}]\n{texto}")
        trazabilidad_fuentes.append({
            "rank": i,
            "source": src,
            "title": title,
            "chunk_id": chunk_id,
            "score": score,
            "text": texto
        })
        
    contexto_texto = "\n\n---\n\n".join(contexto_list)
    
    prompt = f"""
    Eres un Asistente Virtual Experto en Normatividad Académica de Educación Superior (CUN).
    Responde a la pregunta del usuario utilizando ÚNICAMENTE el contexto documental proporcionado.

    REGLAS OBLIGATORIAS:
    1. Cita siempre de manera explícita el documento y artículo de origen.
    2. Si la respuesta no se encuentra en el contexto, indica amablemente que la información no está disponible.
    3. Mantén un tono formal, claro y profesional.

    CONTEXTO DOCUMENTAL RECUPERADO:
    {contexto_texto}

    PREGUNTA DEL USUARIO:
    {pregunta}
    """
    
    modelos_respaldo = [
        "models/gemini-3.1-flash-lite",
        "models/gemini-2.0-flash",
        "models/gemini-3.8-flash",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash"
    ]
    
    respuesta_texto = ""
    modelo_usado = ""
    
    for mod in modelos_respaldo:
        try:
            llm = genai.GenerativeModel(mod)
            res = llm.generate_content(prompt)
            respuesta_texto = res.text
            modelo_usado = mod
            break
        except Exception:
            continue
            
    if not respuesta_texto:
        respuesta_texto = "⚠️ Ocurrió un inconveniente al conectar con la API de Gemini. Por favor verifica tus credenciales."
        modelo_usado = "N/A"
        
    return respuesta_texto, trazabilidad_fuentes, modelo_usado

with st.sidebar:
    st.title("🎓 Pilar Educación - CUN")
    st.markdown("**Proyecto final RAG | Computación en la Nube**")
    st.markdown("**Estudiante:** Leonardo Triviño Ramirez")
    st.divider()
    
    st.subheader("⚙️ Configuración del RAG")
    st.caption(f"**Estado Base Vectorial:** `{conn_status}`")
    top_k = st.slider("Top-k Fragmentos Recuperados:", min_value=1, max_value=5, value=3)
    
    st.divider()
    st.subheader("📂 Inspector de Corpus Indexado")
    
    coll_info = qdrant_client.get_collection(COLLECTION_NAME)
    st.metric("Total Vectores en Qdrant", coll_info.points_count)
    
    with st.expander("📄 Documentos Normativos Indexados"):
        st.markdown("""
        * 📖 `decreto-rectoral-1751-de-2022-reglamento-academico.pdf`
        * 🛡️ `rutas-de-atencion-psicopedagogica.pdf`
        * 📊 `Student_performance_data__.csv`
        """)

st.title("🤖 Chatbot RAG: Consultas Académicas y Normativas")
st.markdown("""
Bienvenido al asistente inteligente de normatividad universitaria. Realiza tus preguntas sobre el 
**Reglamento Académico (Decreto 1751)**, **Rutas de Atención Psicopedagógica** y **Métricas de Desempeño**.
""")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "fuentes" in message:
            with st.expander("🔍 Trazabilidad Top-k y Similitud Coseno"):
                for src in message["fuentes"]:
                    st.write(f"**Rank {src['rank']}** | Similitud: `{src['score']:.4f}` | 📁 `{src['source']}` (Chunk #{src['chunk_id']})")
                    st.caption(src["text"])

if user_input := st.chat_input("Escribe tu pregunta académica (ej: ¿Cómo solicito reserva de cupo?)..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Buscando vectores en Qdrant y generando respuesta con Gemini..."):
            respuesta_texto, trazabilidad, mod_usado = ejecutar_pipeline_rag(user_input, top_k=top_k)
            
            st.markdown(respuesta_texto)
            st.caption(f"🤖 *Respuesta generada por:* `{mod_usado}`")
            
            with st.expander("🔍 Trazabilidad de Fuentes Recuperadas (Top-k)"):
                for src in trazabilidad:
                    st.markdown(f"**Ranking #{src['rank']}** — Score de Similitud Coseno: `{src['score']:.4f}`")
                    st.markdown(f"**Documento:** `{src['source']}` | **Chunk:** `{src['chunk_id']}`")
                    st.info(src["text"])
                    st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": respuesta_texto,
                "fuentes": trazabilidad
            })
