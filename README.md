## 🤖 Bank AI Customer Support Prompt Chain

## 🌟 Project Overview

This repository contains the implementation of a 5-stage Prompt Chain designed to power the initial customer service layer for a bank. The system is built to process a customer's raw, free-text query and systematically guide it through a classification and analysis pipeline, culminating in a categorized, actionable response.

The implementation utilizes the Google Gemini API (gemini-2.5-flash) via the google-genai SDK, demonstrating sequential LLM calls where each stage's output provides critical context for the next.

### 💡 The 5-Stage Prompt Chain

The chain is a logical reasoning pipeline that transforms a qualitative query into quantitative, structured data before generating the final message.

### 🛠️ Setup and Installation
1. Prerequisites
```Python 3.8+```

A Gemini API Key (Obtained from Google AI Studio).

2. Install Dependencies
Install the Google GenAI SDK and python-dotenv to manage your environment variables securely.

```
pip install google-genai python-dotenv
```

3. Configure API Key
Create a file named .env in the root directory of this project.

Add your API key to the file in the exact following format:

```
GEMINI_API_KEY="use your api key"
```

### 🚀 Usage
The core logic is contained within the prompt-chain.py script.

Running the Example
The file contains an if __name__ == "__main__": block that executes a test query and prints the output of all 5 stages.

```
python prompt-chain.py
```