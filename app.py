from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import time
import logging
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import yaml
import os
import os
import asyncio
from IndexManager import IndexManager
from config import retry_limit_var, retry_count_var, wait_duration_var

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("Please set the environment variable OPENAI_API_KEY")

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# This defines an endpoint at the root URL ("/"). When this URL is accessed via a GET request, the function returns a JSON response.
@app.get("/")
async def read_root():
    return {"Hello": "Welcome to Policy Search Engine developed by Simpplr!"}

#This is a Pydantic model which is used to validate the data received from the client. 
#Here, the Query class expects to receive a JSON object with a key text
class Query(BaseModel):
    text: str

#This function loads a YAML configuration file into a Python dictionary.
def load_config(config_path='config.yaml'):
    """ Load configuration from a YAML file. """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Initialize configuration and index at startup
config = load_config()
pdfs_dir = config['paths']['pdfs_dir']
elasticsearch_endpoint_url = os.getenv('ELASTICSEARCH_URL', config['elasticsearch']['endpoint_url'])
index_name = config['elasticsearch']['index_name']
query_engine = None  # This will be initialized when the server starts


# This asynchronous function is intended to be called when the FastAPI application starts. 
# It checks if Elasticsearch is ready and sets up the query_engine once Elasticsearch is available.
async def on_startup():
    global query_engine # why do we need to use global here? Because we are modifying the global variable query_engine inside the function.
    es_client = Elasticsearch(elasticsearch_endpoint_url)

    # Attempt to connect to Elasticsearch with retries
    retry_limit = retry_limit_var
    retry_count = retry_count_var
    wait_duration = wait_duration_var
    while retry_count < retry_limit:
        try:
            if es_client.ping():
                print("Elasticsearch is ready!")
                break
        except Exception as e:
            retry_count += 1
            print(f"Waiting for Elasticsearch to be ready (Attempt {retry_count}/{retry_limit})...")
            time.sleep(wait_duration)  # Wait for 10 seconds before retrying
        if retry_count == retry_limit:
            raise Exception("Failed to connect to Elasticsearch after several retries.")

    manager = IndexManager(pdfs_dir, elasticsearch_endpoint_url, index_name)
    # The await keyword below is used to pause the execution of the containing asynchronous function (on_startup() in this case) until the awaited task (create_query_engine()) is completed. 
    # The use of await here indicates that the create_query_engine() method returns a coroutine, an object that encapsulates the execution of the asynchronous operation, which needs to be resolved (or completed) to continue. 
    # This ensures that the query engine is fully initialized and ready to use before any queries are processed.
    # This line is crucial for ensuring that the query engine is ready and operational before the API starts serving queries. 
    # By awaiting the creation of the query engine, you ensure that all necessary setup tasks (like connecting to Elasticsearch, configuring the index, loading necessary data or schemas, etc.) are complete. 
    query_engine = await manager.create_query_engine()
# This endpoint handles POST requests where a user sends a query in JSON format. The function checks if the query is properly formed, 
# sends it to Elasticsearch (via query_engine), and formats the response before sending it back to the user.
@app.post("/query/")
async def perform_query(query: Query):
    if not query_engine:
        raise HTTPException(status_code=503, detail="Index is not initialized yet")
    if (not query.text) or query.text.strip() == "":
        raise HTTPException(status_code=400, detail="Query text is required")
    try:
        #  check if it's awaitable
        query_response = query_engine.query(query.text)
        if asyncio.iscoroutine(query_response): 
            query_response = await query_response  # Only await if it's actually a coroutine

        formatted_response = IndexManager.simple_format_response_and_sources(query_response)
        return formatted_response
    except Exception as e:
        logger.error(f"An error occurred while processing the query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Attach event handlers
app.router.add_event_handler("startup", on_startup)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
