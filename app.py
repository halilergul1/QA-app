from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import time
import logging
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import yaml  
from utils import simple_format_response_and_sources
import os
import os
import asyncio
from utils import async_load_or_initialize_index


# Load environment variables from .env file
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("Please set the environment variable OPENAI_API_KEY")

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "Welcome to Policy Search Engine of Simpplr!"}

class Query(BaseModel):
    text: str

def load_config(config_path='config.yaml'):
    """ Load configuration from a YAML file. """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Initialize configuration and index at startup
config = load_config()
pdfs_dir = config['paths']['pdfs_dir']
#elasticsearch_endpoint_url = config['elasticsearch']['endpoint_url']
elasticsearch_endpoint_url = os.getenv('ELASTICSEARCH_URL', config['elasticsearch']['endpoint_url'])
index_name = config['elasticsearch']['index_name']
query_engine = None  # This will be initialized when the server starts


async def on_startup():
    global query_engine
    es_client = Elasticsearch(elasticsearch_endpoint_url)

    # Attempt to connect to Elasticsearch with retries
    retry_limit = 10
    retry_count = 0
    while retry_count < retry_limit:
        try:
            if es_client.ping():
                print("Elasticsearch is ready!")
                break
        except Exception as e:
            retry_count += 1
            print(f"Waiting for Elasticsearch to be ready (Attempt {retry_count}/{retry_limit})...")
            time.sleep(10)  # Wait for 10 seconds before retrying
        if retry_count == retry_limit:
            raise Exception("Failed to connect to Elasticsearch after several retries.")

    query_engine = await async_load_or_initialize_index(pdfs_dir, elasticsearch_endpoint_url, index_name)

@app.post("/query/")
async def perform_query(query: Query):
    if not query_engine:
        raise HTTPException(status_code=503, detail="Index is not initialized yet")
    try:
        # Directly call the query without awaiting, but check if it's awaitable
        query_response = query_engine.query(query.text)
        if asyncio.iscoroutine(query_response):
            query_response = await query_response  # Only await if it's actually a coroutine

        formatted_response = simple_format_response_and_sources(query_response)
        return formatted_response
    except Exception as e:
        logger.error(f"An error occurred while processing the query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Attach event handlers
app.router.add_event_handler("startup", on_startup)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
