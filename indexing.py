import os
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.embeddings.openai import OpenAIEmbedding
from ingest import nodes # nodes is the output of ingest.py




elasticsearch_endpoint_url = "http://localhost:9201"  #Elasticsearch endpoint URL
index_name = "semantic-chunk-v5"  # The name of the Elasticsearch index you want to use or create

# Setup for the Elasticsearch vector store
vector_store = ElasticsearchStore(
    index_name=index_name, 
    es_url=elasticsearch_endpoint_url,
)

# Setup the storage context with the vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# setup the index/query process, ie the embedding model (and completion if used)
embed_model = OpenAIEmbedding(
    model='text-embedding-3-small', 
    embed_batch_size=100) # That means you send less requests to the API. So it's faster and cheaper. Netrowk latency is the bottleneck here.

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context
)
index.storage_context.persist(persist_dir="<persist_dir>") # this is the directory where the index will be stored

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)
# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.70)],
)