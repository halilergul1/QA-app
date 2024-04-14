import sys
sys.path.append('/Users/halilergul/Desktop/simpplr-v3')

from llama_index.llms.openai import OpenAI
import os
import os
from llama_index.core import (
    VectorStoreIndex,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from utils import print_response_and_sources


os.environ['OPENAI_API_KEY'] = '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' 
gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo")

elasticsearch_endpoint_url = "http://localhost:9201"  # Or your actual Elasticsearch endpoint URL
index_name = "simpplr-policy-qa"  # The name of the Elasticsearch index you want to use or create

# Setup for the Elasticsearch vector store. This is where the vectors will be stored.
vector_store = ElasticsearchStore(
    index_name=index_name, 
    es_url=elasticsearch_endpoint_url,
)

index = VectorStoreIndex.from_vector_store(
    vector_store #storage_context=storage_context
)
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

response = query_engine.query("What are conditions for remote work?")
print(print_response_and_sources(response))
