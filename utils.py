import string
import asyncio
from elasticsearch import Elasticsearch
from llama_index.core import StorageContext
from elasticsearch import AsyncElasticsearch
import os
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from concurrent.futures import ThreadPoolExecutor



# function to query the index and print the response along with the sources
def print_response_and_sources(response):
    # First, print the response object directly
    print("Response:")
    print(response)
    print("\nSources:")
    source_counter = 1
    # Check if the response object has the attribute 'source_nodes' to avoid errors
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            # Access the metadata for the current node
            metadata = node.node.metadata
            file_name = metadata.get('file_name', 'N/A')  # Default to 'N/A' if not found
            page_label = metadata.get('page_label', 'N/A')  # Default to 'N/A' if not found
            formatted_text = node.node.text.replace('\n\n', '\u2028').replace('\n', ' ').replace('\u2028', '\n\n')
            print(f"Source{source_counter} (File: {file_name}, Page: {page_label}):")
            print(formatted_text)
            print("\n")  # Add extra newline for better separation between sources
            source_counter += 1
    else:
        print("No source nodes found in response.")

def format_response_and_sources(response):
    output = {"response": response, "sources": []}
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            metadata = node.node.metadata
            file_name = metadata.get('file_name', 'N/A')
            page_label = metadata.get('page_label', 'N/A')
            formatted_text = node.node.text.replace('\n\n', '\u2028').replace('\n', ' ').replace('\u2028', '\n\n')
            source_info = {
                "file": file_name,
                "page": page_label,
                "text": formatted_text
            }
            output["sources"].append(source_info)
    else:
        output["sources"] = "No source nodes found in response."
    return output


def simple_format_response_and_sources(response):
    # Check if the response has an attribute 'response', and get its value
    primary_response = getattr(response, 'response', '')

    # Create the output dictionary with the primary response
    output = {"response": primary_response}

    # Check if the response has an attribute 'source_nodes' and process it
    sources = []
    if hasattr(response, 'source_nodes'):
        for node in response.source_nodes:
            # Assume 'node' is an object and access its attributes directly
            node_data = getattr(node, 'node', None)
            if node_data:
                metadata = getattr(node_data, 'metadata', {})
                text = getattr(node_data, 'text', '').replace('\n\n', '\u2028').replace('\n', ' ').replace('\u2028', '\n\n')
                source_info = {
                    "file": metadata.get('file_name', 'N/A'),
                    "page": metadata.get('page_label', 'N/A'),
                    "text": text
                }
                sources.append(source_info)

    # Append the sources list to the output dictionary
    output['sources'] = sources

    return output


from elasticsearch import Elasticsearch

def load_or_initialize_index(pdfs_dir, elasticsearch_endpoint_url, index_name):
    # Create an ElasticsearchStore instance
    vector_store = ElasticsearchStore(index_name=index_name, es_url=elasticsearch_endpoint_url)
    
    # Create Elasticsearch client to check index existence
    es_client = Elasticsearch(elasticsearch_endpoint_url)
    
    # Check if the index exists
    if es_client.indices.exists(index=index_name):
        print(f"Loading existing index '{index_name}' from Elasticsearch...")
        index = VectorStoreIndex.from_vector_store(vector_store)
    else:
        # Index does not exist, so create a new one
        print(f"No existing index found. Initializing new index with name: {index_name}...")
        documents = SimpleDirectoryReader(pdfs_dir).load_data()
        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )
        nodes = splitter.get_nodes_from_documents(documents)
        # Setup the storage context with the vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes,storage_context=storage_context)
        print("New index initialized and persisted.")
    
    # Configure the retriever and response synthesizer
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3
    )
    response_synthesizer = get_response_synthesizer()
    
    # Assemble the query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.70)],
    )
    
    return query_engine


async def async_load_or_initialize_index(pdfs_dir, elasticsearch_endpoint_url, index_name):
    vector_store = ElasticsearchStore(index_name=index_name, es_url=elasticsearch_endpoint_url)
    es_client = AsyncElasticsearch(elasticsearch_endpoint_url)
    
    if await es_client.indices.exists(index=index_name):
        print(f"Loading existing index '{index_name}' from Elasticsearch...")
        index = VectorStoreIndex.from_vector_store(vector_store)
    else:
        print(f"No existing index found. Initializing new index with name: {index_name}...")

        def load_data_sync():
            return SimpleDirectoryReader(pdfs_dir).load_data()

        loop = asyncio.get_running_loop()
        documents = await loop.run_in_executor(None, load_data_sync)

        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"Extracted {len(nodes)} nodes from the documents with number of pages: {len(documents)}")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        print("New index initialized and persisted.")
    
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    response_synthesizer = get_response_synthesizer()
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.70)]
    )
    
    return query_engine



def setup_query_engine(index):
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    response_synthesizer = get_response_synthesizer()
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.70)],
    )