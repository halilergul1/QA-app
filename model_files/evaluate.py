import os
import pandas as pd
from IPython.display import display
from llama_index.core import (SimpleDirectoryReader, StorageContext, Settings)
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import get_response_synthesizer
from dotenv import load_dotenv
from model_files.indexing import embed_model, retriever
from config import semantic_buffer_size, semantic_breakpoint_percentile_threshold, gpt_model, cutoff, ideal_temperature

load_dotenv()  # environment variables

try:
    # Load documents
    documents = SimpleDirectoryReader("pdfs").load_data()
except Exception as e:
    print(f"Error loading documents: {e}")
    documents = []

if documents:
    try:
        # Set up LLM and node parser in settings
        Settings.llm = OpenAI(model=gpt_model)
        Settings.node_parser = SemanticSplitterNodeParser(
            buffer_size=semantic_buffer_size, breakpoint_percentile_threshold=semantic_breakpoint_percentile_threshold, embed_model=embed_model
        )
        Settings.num_output = 512
        Settings.context_window = 3900

        # Generate questions from documents
        data_generator = DatasetGenerator.from_documents(documents)
        eval_questions = data_generator.generate_questions_from_nodes()
    except Exception as e:
        print(f"Error during question generation setup: {e}")
        eval_questions = []

    if eval_questions:
        try:
            # Check for API key
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("Please set the environment variable OPENAI_API_KEY")
            gpt3 = OpenAI(temperature=ideal_temperature, model=gpt_model)
        except Exception as e:
            print(f"Error setting up OpenAI LLM: {e}")
            gpt3 = None

        if gpt3:
            try:
                # Rebuild storage context and configure the response synthesizer
                storage_context = StorageContext.from_defaults(persist_dir="<persist_dir>")
                response_synthesizer = get_response_synthesizer()
                query_engine = RetrieverQueryEngine(
                    retriever=retriever,
                    response_synthesizer=response_synthesizer,
                    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=cutoff)]
                )
                evaluator = RelevancyEvaluator(llm=gpt3)
                eval_data = []
                for question in eval_questions:
                    try:
                        response_vector = query_engine.query(question)
                        eval_result = evaluator.evaluate_response(
                            query=question, response=response_vector
                        )
                        current_eval_data = {
                            "Query": question,
                            "Response": str(response_vector),
                            "Source": response_vector.source_nodes[0].node.text[:1000] + "...",
                            "Evaluation Result": "Pass" if eval_result.passing else "Fail",
                            "Reasoning": eval_result.feedback,
                        }
                        eval_data.append(current_eval_data)

                        # Display the current evaluation result
                        current_eval_df = pd.DataFrame([current_eval_data])
                        display(current_eval_df)
                    except Exception as e:
                        print(f"Error during querying or evaluation: {e}")

                # Save results to CSV
                final_eval_df = pd.DataFrame(eval_data)
                csv_filename = "evaluation_results.csv"
                final_eval_df.to_csv(csv_filename, index=False)
                print(f"Saved evaluation results to {csv_filename}")

            except Exception as e:
                print(f"Error during indexing or evaluation setup: {e}")
else:
    print("No documents or questions available for processing.")
