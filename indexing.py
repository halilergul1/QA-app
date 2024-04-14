import os
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.embeddings.openai import OpenAIEmbedding
from ingest import nodes  # nodes is the output of ingest.py
from config import (
    embedding_model,
    embedding_batch_size,
    top_k_similar,
    cutoff,
)

try:
    elasticsearch_endpoint_url = "http://localhost:9201"  # Elasticsearch endpoint URL
    index_name = "semantic-chunk-v5"  # The name of the Elasticsearch index you want to use or create

    # Setup for the Elasticsearch vector store
    vector_store = ElasticsearchStore(
        index_name=index_name,
        es_url=elasticsearch_endpoint_url,
    )
except Exception as e:
    print(f"Failed to setup Elasticsearch vector store: {e}")
    vector_store = None

# Proceed only if vector store is successfully set up
if vector_store:
    try:
        # Setup the storage context with the vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    except Exception as e:
        print(f"Failed to setup storage context: {e}")
        storage_context = None

    if storage_context:
        try:
            # Setup the index/query process, i.e., the embedding model
            embed_model = OpenAIEmbedding(
                model=embedding_model,
                embed_batch_size=embedding_batch_size
            )
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context
            )
            # Persist the index
            index.storage_context.persist(persist_dir="<persist_dir>")  # this is the directory where the index will be stored
        except Exception as e:
            print(f"Failed to setup index or persist data: {e}")
            index = None

        if index:
            try:
                # Configure retriever
                retriever = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=top_k_similar,
                )
                # Configure response synthesizer
                response_synthesizer = get_response_synthesizer()

                # Assemble query engine
                query_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=cutoff)],
                )
            except Exception as e:
                print(f"Failed to configure components of the query engine: {e}")
else:
    print("Vector store setup failed, skipping all dependent steps.")