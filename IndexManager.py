import string
import re
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
from concurrent.futures import ThreadPoolExecutor
from config import semantic_buffer_size, semantic_breakpoint_percentile_threshold, embedding_model, top_k_similar, cutoff

class IndexManager:
    def __init__(self, pdfs_dir, elasticsearch_endpoint_url, index_name):
        self.pdfs_dir = pdfs_dir
        self.elasticsearch_endpoint_url = elasticsearch_endpoint_url
        self.index_name = index_name
        self.es_client = AsyncElasticsearch(elasticsearch_endpoint_url)
        self.vector_store = ElasticsearchStore(index_name=self.index_name, es_url=self.elasticsearch_endpoint_url)

    async def check_index_exists(self):
        return await self.es_client.indices.exists(index=self.index_name)

    async def load_data(self):
        def load_data_sync():
            return SimpleDirectoryReader(self.pdfs_dir).load_data()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, load_data_sync)

    async def process_documents(self, documents):
        embed_model = OpenAIEmbedding(model=embedding_model)
        splitter = SemanticSplitterNodeParser(
            buffer_size=semantic_buffer_size, breakpoint_percentile_threshold=semantic_breakpoint_percentile_threshold, embed_model=embed_model
        )
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"Extracted {len(nodes)} nodes from the documents with number of pages: {len(documents)}")
        return nodes

    async def initialize_index(self, nodes):
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        print("New index initialized and persisted.")
        return index

    async def get_or_create_index(self):
        if await self.check_index_exists():
            print(f"Loading existing index '{self.index_name}' from Elasticsearch...")
            return VectorStoreIndex.from_vector_store(self.vector_store)
        else:
            print(f"No existing index found. Initializing new index with name: {self.index_name}...")
            documents = await self.load_data()
            nodes = await self.process_documents(documents)
            return await self.initialize_index(nodes)

    async def create_query_engine(self):
        index = await self.get_or_create_index()
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k_similar)
        response_synthesizer = get_response_synthesizer()
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=cutoff)]
        )
        return query_engine
    
    @staticmethod
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
                    text = getattr(node_data, 'text', '')
                    text = re.sub(r'\n\n|\n|\u2028', lambda m: {'\n\n': '\u2028', '\n': ' ', '\u2028': '\n\n'}[m.group()], text)
                    source_info = {
                        "file": metadata.get('file_name', 'N/A'),
                        "page": metadata.get('page_label', 'N/A'),
                        "text": text
                    }
                    sources.append(source_info)

        # Append the sources list to the output dictionary
        output['sources'] = sources

        return output
