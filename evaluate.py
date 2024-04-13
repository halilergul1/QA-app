import os
from IPython.display import display
import pandas as pd
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import get_response_synthesizer
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import (SimpleDirectoryReader, StorageContext)
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from indexing import embed_model # embed_model is a variable in indexing.py
from indexing import retriever # retriever is a variable in indexing.py
from dotenv import load_dotenv


# question generation

# Load documents
documents = SimpleDirectoryReader("pdfs").load_data()

#set up llm model
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.node_parser = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)
Settings.num_output = 512
Settings.context_window = 3900

#generate questions
data_generator = DatasetGenerator.from_documents(documents)
eval_questions = data_generator.generate_questions_from_nodes()

# Load environment variables from .env file if present
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("Please set the environment variable OPENAI_API_KEY")
gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>") # also load_index_from_storage might be used

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

#now query the index
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.70)]
)

# define evaluator
evaluator = RelevancyEvaluator(llm=gpt3)


import pandas as pd

# Initialize a list to hold all the evaluation data
eval_data = []

# Loop through all questions in eval_questions
for question in eval_questions:
    # Query the index
    response_vector = query_engine.query(question)
    
    # Evaluate the response
    eval_result = evaluator.evaluate_response(
        query=question, response=response_vector
    )
    
    # Append the relevant data to the eval_data list
    current_eval_data = {
        "Query": question,
        "Response": str(response_vector),
        "Source": response_vector.source_nodes[0].node.text[:1000] + "...",
        "Evaluation Result": "Pass" if eval_result.passing else "Fail",
        "Reasoning": eval_result.feedback,
    }
    eval_data.append(current_eval_data)

    # Create a temporary DataFrame for the current evaluation data
    current_eval_df = pd.DataFrame([current_eval_data])
    
    # Optionally adjust the display properties for better readability
    current_eval_df_styled = current_eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    
    # Display the current evaluation result
    display(current_eval_df_styled)

# Create a DataFrame from the collected data
final_eval_df = pd.DataFrame(eval_data)

# Save the DataFrame to a CSV file
csv_filename = "evaluation_results.csv"
final_eval_df.to_csv(csv_filename, index=False)
print(f"Saved evaluation results to {csv_filename}")
