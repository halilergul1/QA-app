import os
from IndexManager import IndexManager
from indexing import query_engine  # query_engine is the output of indexing.py

try:
    # Execute the query using the query engine
    response = query_engine.query("What are conditions for Paternity Leave?")
except Exception as e:
    print(f"Failed to execute query: {e}")
    response = None

# Proceed only if a response is obtained
if response:
    try:
        # Format and display the response along with its sources
        formatted_response = IndexManager.simple_format_response_and_sources(response)
        print(formatted_response)
    except Exception as e:
        print(f"Failed to format or display the response: {e}")
else:
    print("No response received, skipping formatting and display.")
