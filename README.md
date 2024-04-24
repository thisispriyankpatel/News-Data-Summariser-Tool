# News Data Summariser Tool 📰📰📰

Welcome to the News Data Summariser Tool! This is a Streamlit application designed to simplify the process of researching news articles. Users can input URLs of news articles, ask questions related to the content, and receive summarized information along with relevant sources. This tool utilizes OpenAI's Language Model (LLM) and vector data concepts for text summarization and analysis, providing a robust solution for digesting news content.

## Installation

1. **Clone the Repository:**
    ```sh
    git clone git@github.com:thisispriyankpatel/News-Data-Summariser-Tool-.git
    ```

2. **Navigate to the Directory:**
    ```sh
    cd News-Data-Summariser-Tool-
    ```

3. **Create a Virtual Environment (optional but recommended):**
    ```sh
    python3 -m venv venv
    ```

4. **Activate the Virtual Environment:**
    - Windows:
    ```sh
    venv\Scripts\activate
    ```
    - macOS/Linux:
    ```sh
    source venv/bin/activate
    ```

5. **Install Required Packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To use this tool, follow these steps:

1. **Run the Streamlit App:**
    ```sh
    streamlit run your_script_name.py
    ```

Replace `your_script_name.py` with the name of your Python script containing the provided code.

## Features

- **Input Section:** Allows users to input URLs of news articles for processing.
- **Result Section:** Provides summarized information and relevant sources based on user queries.
- **Language Model Integration:** Utilizes OpenAI's Language Model for text summarization and analysis.
- **Vector Data Concepts:** Implements vector data concepts for efficient document retrieval and analysis.
- **Langchain Integration:** Integrates langchain, a library for natural language processing tasks, including OpenAIEmbeddings for generating embeddings, RetrievalQAWithSourcesChain for question answering with sources, and UnstructuredURLLoader for loading unstructured data from URLs.
- **Interactive UI:** Streamlit-based interface for easy interaction and visualization.

## About Langchain Components

- **OpenAIEmbeddings:** Provides embeddings for text data using OpenAI's language model.
- **RetrievalQAWithSourcesChain:** A chain for question answering with sources, utilizing embeddings and document retrieval techniques.
- **UnstructuredURLLoader:** A component for loading unstructured data from URLs.

## Credits

This project was created by Priyank Patel
