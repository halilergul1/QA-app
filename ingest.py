from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser



# dosya olmayabilir, boş olabilir. O senaryolaroı cover edeceksin.

documents = SimpleDirectoryReader("pdfs").load_data()
print(f"Loaded {len(documents)} document(s).")

embed_model = OpenAIEmbedding(model="text-embedding-3-small",  embed_batch_size=100)
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)
nodes = splitter.get_nodes_from_documents(documents)

print(f"Extracted {len(nodes)} nodes from the documents.")