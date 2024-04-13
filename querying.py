import os
from utils import print_response_and_sources
from indexing import query_engine # query_engine is the output of indexing.py


response = query_engine.query("What are conditions for Paternity Leave?")
print_response_and_sources(response)