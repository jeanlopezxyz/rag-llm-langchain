from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import os

def download_models():
    """
    Descarga y guarda los modelos necesarios en directorios locales.
    """
    # Modelo para embeddings
    embedding_model_name = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    embedding_model_path = os.path.join('models', embedding_model_name)
    
    if not os.path.exists(embedding_model_path):
        print(f"Descargando el modelo de embeddings: {embedding_model_name}...")
        model = SentenceTransformer(embedding_model_name)
        model.save(embedding_model_path)
        print(f"Modelo de embeddings guardado en: {embedding_model_path}")
    else:
        print(f"El modelo de embeddings ya existe en: {embedding_model_path}")

    # Modelo para Cross-Encoder
    cross_encoder_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    cross_encoder_model_path = os.path.join('models', cross_encoder_model_name)
    
    if not os.path.exists(cross_encoder_model_path):
        print(f"Descargando el modelo de Cross-Encoder: {cross_encoder_model_name}...")
        cross_encoder = CrossEncoder(cross_encoder_model_name)
        # La librería CrossEncoder no tiene un método .save() directo,
        # pero al instanciarlo se descarga al caché. La mejor práctica es
        # moverlo a una ruta local, pero por simplicidad, asegurar que el
        # caché esté poblado antes de construir la imagen es suficiente.
        # Para un enfoque más robusto, se usa la descarga manual con la librería `huggingface_hub`.
        print(f"Modelo Cross-Encoder descargado al caché.")
    else:
        print(f"El modelo de Cross-Encoder ya existe en: {cross_encoder_model_path}")


if __name__ == "__main__":
    # Asegurarse que el directorio base 'models' exista
    if not os.path.exists('models'):
        os.makedirs('models')
    download_models()