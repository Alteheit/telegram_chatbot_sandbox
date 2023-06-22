import os

CHROMA_DATA_DIRECTORY: str = os.path.join(os.getcwd(), 'var', 'chroma_data')
UPDATE_ID_CACHE_DURATION_SECONDS: int = 15 * 60

PANELIST_CONTEXTS_DIRECTORY: str = os.path.join(os.getcwd(), 'var', 'contexts', 'panelist')
PANELIST_LANGCHAIN_VECTOR_COLLECTION_NAME: str = "panelist_langchain_store"
PREFECT_CONTEXTS_DIRECTORY: str = os.path.join(os.getcwd(), 'var', 'contexts', 'prefect')
PREFECT_LANGCHAIN_VECTOR_COLLECTION_NAME: str = "prefect_langchain_store"
