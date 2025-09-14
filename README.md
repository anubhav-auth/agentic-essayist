Agentic Essayist
================

Description
-----------

Agentic Essayist is an autonomous research agent and API that specializes in the writings of Paul Graham. It uses a retrieval-augmented generation (RAG) pipeline to answer questions based on a collection of his essays. You can ask it about startups, programming, technology, and other topics covered in his writing, and it will provide a synthesized answer with citations from the text.

This project is a great starting point for anyone looking to build their own agentic AI applications with LangChain, FastAPI, and a vector database like ChromaDB.

Features
--------

-   **FastAPI Backend**: A robust and easy-to-use API for interacting with the agent.

-   **LangChain Agent**: Leverages the power of LangChain to create a ReAct-style agent that can reason and use tools.

-   **ChromaDB Vector Store**: Uses a local ChromaDB vector store to efficiently search and retrieve relevant information from Paul Graham's essays.

-   **Hugging Face Embeddings**: Employs sentence-transformer models from Hugging Face to create vector embeddings of the text.

-   **Data Ingestion Script**: A simple script to process and ingest the text data into the vector store.

How It Works
------------

The project is divided into two main parts: data ingestion and the agentic API.

1.  **Data Ingestion (`ingest.py`)**:

    -   The `ingest.py` script first loads the text from the `data` directory (in this case, `paul_graham_essay.txt`).

    -   It then splits the documents into smaller, more manageable chunks using a `RecursiveCharacterTextSplitter`.

    -   Next, it uses a Hugging Face sentence-transformer model (`all-MiniLM-L6-v2`) to create vector embeddings for each chunk.

    -   Finally, it saves these embeddings to a local ChromaDB vector store in the `chroma` directory.

2.  **Agentic API (`main.py`)**:

    -   The `main.py` script sets up a FastAPI application with a single endpoint: `/ask_agent`.

    -   It defines a `search_paul_graham_essays` tool that the agent can use. This tool queries the ChromaDB vector store to find the most relevant chunks of text based on the user's question.

    -   A ReAct-style agent is created using LangChain's `create_react_agent`. This agent is given a prompt that instructs it to act as a Paul Graham expert and to use the search tool to answer questions.

    -   When a question is sent to the `/ask_agent` endpoint, the agent executor invokes the agent, which then uses the search tool to find relevant information and synthesizes a final answer.

Project Structure
-----------------

```
.
├── chroma/                 # Directory for the ChromaDB vector store
├── data/
│   └── paul_graham_essay.txt # The text data to be ingested
├── .gitignore              # Files to be ignored by Git
├── ingest.py               # Script for data ingestion
├── main.py                 # The main FastAPI application
└── requirements.txt        # (You'll need to create this)

```

Setup and Installation
----------------------

1.  **Clone the repository:**

    Bash

    ```
    git clone https://github.com/your-username/agentic-essayist.git
    cd agentic-essayist

    ```

2.  **Create a virtual environment:**

    Bash

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    ```

3.  Install the dependencies:

    You'll need to create a requirements.txt file with the following content:

    ```
    fastapi
    uvicorn
    langchain
    langchain-google-genai
    langchain-community
    chromadb
    sentence-transformers
    python-dotenv

    ```

    Then, install the dependencies:

    Bash

    ```
    pip install -r requirements.txt

    ```

4.  Set up your Google API Key:

    Create a .env file in the root of the project and add your Google API key:

    ```
    GOOGLE_API_KEY="your-google-api-key"

    ```

5.  Run the data ingestion script:

    This will create the ChromaDB vector store in the chroma directory.

    Bash

    ```
    python ingest.py

    ```

Usage
-----

1.  **Start the FastAPI server:**

    Bash

    ```
    uvicorn main:app --reload

    ```

2.  Access the API documentation:

    Once the server is running, you can access the interactive API documentation at http://127.0.0.1:8000/docs.

3.  Send a request:

    You can use the API documentation to send a POST request to the /ask_agent endpoint with a JSON body like this:

    JSON

    ```
    {
      "question": "What did Paul Graham work on before college?"
    }

    ```

    Or, you can use a tool like `curl`:

    Bash

    ```
    curl -X POST "http://127.0.0.1:8000/ask_agent" -H "Content-Type: application/json" -d '{"question": "What did Paul Graham work on before college?"}'

    ```

API Endpoint
------------

### `POST /ask_agent`

-   **Request Body:**

    -   `question` (string, required): The question to ask the agent.

-   **Response Body:**

    -   `answer` (string): The agent's answer to the question.

Dependencies
------------

-   **fastapi**: For creating the API.

-   **uvicorn**: For running the FastAPI server.

-   **langchain**: The main framework for building the agent.

-   **langchain-google-genai**: For using Google's generative models.

-   **langchain-community**: For community-provided LangChain components.

-   **chromadb**: The vector store for storing and retrieving embeddings.

-   **sentence-transformers**: For creating the text embeddings.

-   **python-dotenv**: For loading environment variables.

Future Improvements
-------------------

-   **Deploy to a cloud service**: Deploy the application to a cloud service like Google Cloud Run or AWS Lambda for a production-ready setup.

-   **Add more data sources**: Expand the knowledge base of the agent by adding more of Paul Graham's essays or other relevant documents.

-   **Implement a more sophisticated agent**: Experiment with different agent types and models to improve the agent's reasoning and response generation capabilities.

-   **Add a frontend**: Create a simple frontend with a chat interface to make the agent more user-friendly.
