from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from config import semantic_buffer_size, embedding_model

try:
    # Load documents
    documents = SimpleDirectoryReader("pdfs").load_data()
    print(f"Loaded {len(documents)} document(s).")
except Exception as e:
    print(f"Failed to load documents: {e}")
    documents = []

# Proceed only if documents are successfully loaded
if documents:
    try:
        # Initialize the embedding model
        embed_model = OpenAIEmbedding(model=embedding_model, embed_batch_size=100)
    except Exception as e:
        print(f"Failed to initialize embedding model: {e}")
        embed_model = None

    if embed_model:
        try:
            # Initialize the semantic splitter with the embedding model
            splitter = SemanticSplitterNodeParser(
                buffer_size=semantic_buffer_size, breakpoint_percentile_threshold=95, embed_model=embed_model
            )
            # Extract nodes from documents
            nodes = splitter.get_nodes_from_documents(documents)
            print(f"Extracted {len(nodes)} nodes from the documents.")
        except Exception as e:
            print(f"Failed to extract nodes from documents: {e}")
else:
    print("No documents loaded, skipping processing.")
