import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from a .env file
load_dotenv()

# --- 1. SETUP THE RETRIEVER ---

CHROMA_PATH = "chroma"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@tool
def search_paul_graham_essays(query: str) -> str:
    """
    Searches and returns relevant information from a collection of Paul Graham's essays.
    Use this tool to answer questions about Paul Graham's writings, startups, programming, and related tech topics.
    """
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # --- FIX 1: Use similarity_search instead of similarity_search_with_relevance_scores ---
    results = db.similarity_search(query, k=3)

    if not results:
        return "I could not find any relevant information in Paul Graham's essays for that query."
    
    # We no longer need to check the score, just format the content.
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    return context_text

# --- 2. SETUP THE AGENT ---

REACT_PROMPT_TEMPLATE = """
You are an expert researcher specializing in the writings of Paul Graham. Your goal is to provide accurate and comprehensive answers to user questions based on his essays.

To do this, you MUST use the `search_paul_graham_essays` tool to find relevant quotes and ideas.

- You may need to use the tool multiple times to gather enough information for complex questions.
- After your research is complete, synthesize the information from the tool to provide a final, coherent answer.
- In your final answer, cite the specific ideas you are referencing from your search results.

TOOLS:
------
You have access to the following tools:
{tools}

To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
```

When you have a response to say to the user, or if you do not need to use a tool, you MUST use the following format:
```
Thought: Do I need to use a tool? No
Final Answer: [Your comprehensive and well-structured response here]
```

Begin!

New input: {input}
{agent_scratchpad}
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
tools = [search_paul_graham_essays]
agent = create_react_agent(llm, tools, prompt)

# --- FIX 2: Add handle_parsing_errors=True to make the agent more robust ---
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True # This is the safety net
)


# --- 3. SETUP THE FASTAPI APP ---

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the agent.")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The agent's answer to the question.")

app = FastAPI(
    title="Agentic Essayist API",
    description="An API for querying the writings of Paul Graham using an autonomous research agent.",
    version="1.0.0",
)

@app.post("/ask_agent", response_model=QueryResponse)
async def ask_agent_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Receives a question, invokes the agent executor, and returns the synthesized answer.
    """
    response = await agent_executor.ainvoke({"input": request.question})
    return QueryResponse(answer=response["output"])

