import os
import sys
sys.path.append(os.getcwd())

import config
import secret_config
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader

if __name__ == '__main__':
    # Bootstrapping
    os.environ['OPENAI_API_KEY'] = secret_config.OPENAI_API_KEY
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
    )
    # Generate documents
    document_paths = [
        os.path.join(config.PANELIST_CONTEXTS_DIRECTORY, file_name)
        for file_name in os.listdir(config.PANELIST_CONTEXTS_DIRECTORY)
    ]
    loaders = [
        TextLoader(document_path) for document_path in document_paths
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    texts = text_splitter.split_documents(documents)
    # Save to PGVector
    db = PGVector.from_documents(
        texts,
        embeddings,
        collection_name=config.PANELIST_LANGCHAIN_VECTOR_COLLECTION_NAME,
        connection_string=secret_config.PANELIST_PGVECTOR_CONNECTION_STRING,
    )
