from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from dotenv import load_dotenv
import yaml


def load_config(config_path='config.yaml'):
    """ Load configuration from a YAML file. """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Initialize configuration and index at startup
config = load_config()

elasticsearch_endpoint_url = "http://localhost:9201"  # Or your actual Elasticsearch endpoint URL
index_name = "simpplr-policy-qa"  # The name of the Elasticsearch index you want to use or create

vector_store = ElasticsearchStore(
    index_name=index_name, 
    es_url=elasticsearch_endpoint_url,
)

from elasticsearch import Elasticsearch
es = Elasticsearch(elasticsearch_endpoint_url)
es.info()


if es.indices.exists(index=index_name): # this is not working for NOW @TODO
    print("Loading existing index from Elasticsearch...")
else:
    print("Creating new index in Elasticsearch...")