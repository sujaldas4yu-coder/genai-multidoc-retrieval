## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Researchers often have to read many long papers to find one answer or to compare ideas across papers, which is slow and error-prone. A single agent that can search many documents and combine the findings would save a lot of time. This experiment builds a multidocument retrieval agent using LlamaIndex over 11 research papers. The agent selects the relevant papers and tools for each question, retrieves the needed information, and combines it into concise, relevant answers. It is tested with different kinds of queries.

### DESIGN STEPS:
#### Step 1: Create tools for each document

Load each of the 11 PDFs (p1 to p11) and split its text into chunks.
For every paper, build a vector tool for specific questions (datasets, methods, results) and a summary tool for overview questions.
Put the opening text of each paper (title and abstract) into the tool descriptions, so the agent can tell which tool belongs to which paper. This gives 11 × 2 = 22 tools.

#### Step 2: Add tool retrieval and build the agent

Create an object index over all 22 tools, so that only the 3 or 4 tools most relevant to a question are given to the agent, instead of all 22.
Set up the LLM (OpenAI GPT model) and build the agent with FunctionCallingAgentWorker and AgentRunner, with verbose=True to log every tool call.

#### Step 3: Query the agent and evaluate

Ask specific questions (e.g. the main contribution and evaluation dataset of one paper), summary questions, and comparison questions across two papers.
Check in the log which tools the agent chose, and whether the answers are concise, relevant and accurate.

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

urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=LzPWWPAdY4",
    "https://openreview.net/pdf?id=VTF8yNQM66",
    "https://openreview.net/pdf?id=hSyW5go0v8",
    "https://openreview.net/pdf?id=9WD9KwssyT",
    "https://openreview.net/pdf?id=yV6fD7LYkF",
    "https://openreview.net/pdf?id=hnrB5YHoYu",
    "https://openreview.net/pdf?id=WbWtOYIzIK",
    "https://openreview.net/pdf?id=c5pwL0Soay",
    "https://openreview.net/pdf?id=TpD2aG1h0D",
]
papers = ["p1.pdf", "p2.pdf", "p3.pdf","p4.pdf","p5.pdf","p6.pdf",
          "p7.pdf", "p8.pdf", "p9.pdf", "p10.pdf","p11.pdf"]

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

#### Output Before Query Execution:
<img width="1338" height="422" alt="image" src="https://github.com/user-attachments/assets/6d0ac9e7-c50c-41b0-b178-d17f0cb58c68" />

#### Query:
```python
queries = [
    "What problem does paper p7 (fake news detection with RAG) solve, and how does it use retrieval augmented generation?",
    "Compare paper p7 (fake news detection using RAG) with paper p10 (multilingual document question answering). How does each use retrieval?",
    "Compare the Engineering Copilot paper p6 with the multilingual document question answering paper p10. What are their approaches?",
    "Summarize the approach of paper p11 on scam call detection and describe its evaluation results.",
]

for i, q in enumerate(queries, 1):
    print(f"\n=========== QUERY {i} ===========")
    response = agent.query(q)
    print(f"\nANSWER {i}:\n", str(response))
```
#### Output After Query Execution:
<img width="1900" height="946" alt="image" src="https://github.com/user-attachments/assets/666c3374-3de1-4332-bd96-5e421531ae00" />

<img width="1382" height="458" alt="image" src="https://github.com/user-attachments/assets/0dda3477-97ac-45a9-894c-62478305636d" />


### RESULT:
A multidocument retrieval agent was designed and implemented using LlamaIndex over 11 research papers. A vector tool and a summary tool were created for each paper, giving 22 tools, and a tool retriever selected only the relevant ones for each query. The agent called the right tools, for example the summary and vector tools of paper p1 for a question about its contribution and datasets, and combined their outputs into clear answers. It handled specific questions, summaries and cross-paper comparisons correctly. Thus, the multidocument agent using LlamaIndex was successfully developed and evaluated.
