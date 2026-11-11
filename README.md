## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Researchers often have to read many long research papers to find one answer or to compare ideas across papers, which is slow and error-prone. An agent that can search several documents at once and combine the findings would save time. This experiment builds a multidocument retrieval agent using LlamaIndex. The agent reads multiple research PDFs, picks the right tool for each question, and returns concise, relevant answers. It is tested with different kinds of queries.

### DESIGN STEPS:

#### STEP 1: Create tools for each document

Load each research PDF (MAP, newspaper and MERAG) and split its text into chunks.
Build two tools for every paper: a vector tool for specific questions (e.g. dataset, results) and a summary tool for overview questions.
Combine all the tools (3 papers × 2 tools = 6 tools) into one list.

#### STEP 2:Build the agent

Set up the LLM (OpenAI GPT model) and the embedding model.
Create the agent with FunctionCallingAgentWorker and AgentRunner, giving it the list of all tools.
Turn on verbose=True so every tool call the agent makes is shown.

#### STEP 3:Query the agent and evaluate

Ask a specific question (e.g. the evaluation dataset and results of the newspaper paper) and a multi-document question (e.g. a summary of both MERAG and MAP).
Check in the log which tools the agent chose, and whether the answers are concise, relevant and accurate.
Test with different kinds of queries (specific, summary, comparison) to judge how well it handles them.

### PROGRAM:
```python
import sys, os, subprocess, importlib

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "llama-index==0.11.23", "llama-index-llms-openai",
    "llama-index-embeddings-openai", "nest_asyncio", "python-dotenv"])
importlib.invalidate_caches()

from getpass import getpass
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
import nest_asyncio
nest_asyncio.apply()

load_dotenv(find_dotenv())
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

llm = OpenAI(model="gpt-4o-mini")
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

papers = ["MAP.pdf", "newspaper.pdf", "MERAG.pdf"]

missing = [p for p in papers if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(
        f"Missing: {missing}\nFiles in this folder: {os.listdir('.')}"
    )

def get_doc_tools(file_path: str, name: str):
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    nodes = SentenceSplitter(chunk_size=1024).get_nodes_from_documents(documents)

    vector_index = VectorStoreIndex(nodes)

    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        """Answer a specific question about the paper, optionally limited to given page numbers."""
        page_numbers = page_numbers or []
        metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR),
        )
        return query_engine.query(query)

    vector_tool = FunctionTool.from_defaults(name=f"vector_tool_{name}", fn=vector_query)

    summary_index = SummaryIndex(nodes)
    summary_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_engine,
        description=f"Useful for summarization questions related to {name}",
    )
    return vector_tool, summary_tool

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
print("Total tools:", len(initial_tools))

agent_worker = FunctionCallingAgentWorker.from_tools(initial_tools, llm=llm, verbose=True)
agent = AgentRunner(agent_worker)
print("Agent ready")

response = agent.query(
    "Tell me about the evaluation dataset used in newspaper, "
    "and then tell me about the evaluation results"
)
print("\nANSWER 1:\n", str(response))

response = agent.query("Give me a summary of both MERAG and MAP")
print("\nANSWER 2:\n", str(response))

```

### OUTPUT:
<img width="1903" height="1036" alt="Screenshot 2026-11-11 142615" src="https://github.com/user-attachments/assets/e6eaa5c7-9c5a-49bb-ae42-406fa614c8d5" />


### RESULT:
A multidocument retrieval agent was designed and implemented using LlamaIndex. For each of the three research papers, a vector tool and a summary tool were created, and the six tools were given to a function-calling agent with a GPT model. The agent chose the right tool for each question, retrieved the relevant parts of the papers, and combined them into clear answers. It handled specific questions, summaries across two papers and comparison queries correctly. Thus, the multidocument retrieval agent was successfully developed and evaluated.
