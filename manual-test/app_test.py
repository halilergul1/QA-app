import os
from dotenv import load_dotenv
import yaml
from utils import load_or_initialize_index, print_response_and_sources
import logging

# Load environment variables from .env file if present
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("Please set the environment variable OPENAI_API_KEY")

# Set up logging
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path='../config.yaml'):
    """ Load configuration from a YAML file. """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration from YAML
    config = load_config()
    pdfs_dir = config['paths']['pdfs_dir']
    elasticsearch_endpoint_url = config['elasticsearch']['endpoint_url']
    print(elasticsearch_endpoint_url)
    index_name = config['elasticsearch']['index_name']

    # Load or initialize the index without using a persist directory
    query_engine = load_or_initialize_index(pdfs_dir, elasticsearch_endpoint_url, index_name)

    # Example queries
    query = "What is the weather in Istanbul today?"
    # Perform the query
    response = query_engine.query(query)
    #print_response_and_sources(response)
    print(response.source_nodes)

if __name__ == '__main__':
    main()
