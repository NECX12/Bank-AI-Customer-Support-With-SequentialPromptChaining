import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = 'gemini-2.5-flash'

try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure the GEMINI_API_KEY environment variable is set.")
    client = None

AVAILABLE_CATEGORIES = ", ".join([
    "Account Opening", "Billing Issue", "Account Access", 
    "Transaction Inquiry", "Card Services", "Account Statement", 
    "Loan Inquiry", "General Information"
])


def get_prompt_text(stage: int, context: Dict[str, str]) -> str:
    """Returns the full prompt for the given stage, injecting necessary context."""

    if stage == 1:
        return f"""
        System Role: You are an expert customer service analyst. Your task is to interpret a raw customer query and determine the core intent and sentiment.
        
        Task: Analyze the following customer query and provide a **single, brief, and objective summary** of the customer's intent and purpose. Do not attempt to categorize or respond yet.
        
        Customer Query:
        {context['customer_query']}
        
        Output Format Example: The customer wants to know why a charge on their debit card statement is unfamiliar and needs it resolved.
        """

    elif stage == 2:
        # Stage 2: Map the query to possible categories
        return f"""
        System Role: You are a banking system classifier. Your task is to match a summarized customer intent to all relevant service categories.
        
        Available Categories: {AVAILABLE_CATEGORIES}
        
        Task: Given the customer's summarized intent, identify **all possible categories** that could potentially apply. Output the categories as a comma-separated list.
        
        Summarized Intent:
        {context['stage_1_output']}
        
        Output Format Example: Transaction Inquiry, Card Services, Billing Issue
        """

    elif stage == 3:
        # Stage 3: Choose the most appropriate category
        return f"""
        System Role: You are a senior banking service manager. Your task is to review potential categories and select the absolute best fit.
        
        Task: From the list of possible categories provided below, select the **single most appropriate category** that accurately and specifically represents the customer's summarized intent. Output **only the name of the category**.
        
        Summarized Intent:
        {context['stage_1_output']}
        
        Possible Categories:
        {context['stage_2_output']}
        
        Output Format Example: Transaction Inquiry
        """

    elif stage == 4:
        # Stage 4: Extract additional details
        return f"""
        System Role: You are a customer information extractor. Your task is to identify and list any missing or necessary details required to resolve the customer’s query based on the final category chosen.
        
        Task: Based on the **Final Category** and the **Summarized Intent**, identify up to three pieces of **critical, missing information** needed to proceed (e.g., Account number, date, amount, last 4 digits of card, etc.). If no information is needed, state "None". Output the required details as a comma-separated list.
        
        Final Category:
        {context['stage_3_output']}
        
        Summarized Intent (Original Source):
        {context['stage_1_output']}
        
        Output Format Example: Transaction Date, Transaction Amount, Card Type (Debit or Credit)
        """

    elif stage == 5:
        # Stage 5: Generate a short response
        return f"""
        System Role: You are a professional, helpful customer service agent. Your task is to draft a brief and courteous response.
        
        Task: Draft a **short, professional, and empathetic response** to the customer. Acknowledge their issue based on the **Final Category** and **Summarized Intent**, and politely ask them to provide the **Additional Details Needed** to help them immediately. The response should be 1-3 sentences long.
        
        Final Category:
        {context['stage_3_output']}
        
        Summarized Intent:
        {context['stage_1_output']}
        
        Additional Details Needed:
        {context['stage_4_output']}
        
        Output Format Example: Thank you for reaching out regarding your transaction inquiry. We understand this is important. To help us locate the charge and resolve this, could you please provide the Transaction Date, Transaction Amount, and whether it was a Debit or Credit card?
        """
    return "Error: Invalid stage number."

def gemini_llm_call(prompt_text: str) -> str:
    """Calls the Gemini API with the given prompt text."""
    if client is None:
        return "ERROR: Gemini client not initialized. Check API key."

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_text,
            config=types.GenerateContentConfig( 
                temperature=0.0
            )
        )
        
        if response.text and response.text.strip():
            return response.text.strip()
        else:
            return "ERROR in LLM call: Received empty or invalid response text."

    except Exception as e:
        # This handles network/connection errors
        return f"ERROR in LLM call: {e}"

def run_prompt_chain(customer_query: str) -> List[str]:

    # 1. Initialize context and results list
    chain_context: Dict[str, str] = {"customer_query": customer_query}
    results: List[str] = []

    # 2. Execute the stages sequentially

    # Stage 1: Interpret the customer’s intent
    prompt_1 = get_prompt_text(1, chain_context)
    stage_1_output = gemini_llm_call(prompt_1)
    results.append(stage_1_output)
    chain_context["stage_1_output"] = stage_1_output

    # Stage 2: Map the query to possible categories
    prompt_2 = get_prompt_text(2, chain_context)
    stage_2_output = gemini_llm_call(prompt_2)
    results.append(stage_2_output)
    chain_context["stage_2_output"] = stage_2_output

    # Stage 3: Choose the most appropriate category
    prompt_3 = get_prompt_text(3, chain_context)
    stage_3_output = gemini_llm_call(prompt_3)
    results.append(stage_3_output)
    chain_context["stage_3_output"] = stage_3_output

    # Stage 4: Extract additional details
    prompt_4 = get_prompt_text(4, chain_context)
    stage_4_output = gemini_llm_call(prompt_4)
    results.append(stage_4_output)
    chain_context["stage_4_output"] = stage_4_output
    
    # Stage 5: Generate a short response
    prompt_5 = get_prompt_text(5, chain_context)
    stage_5_output = gemini_llm_call(prompt_5)
    results.append(stage_5_output)

    # 3. Return the list of all intermediate results
    return results

if __name__ == "__main__":
    
    print("--- Running Gemini Prompt Chain Example ---")
    
    query = "I cannot log into my online banking account, I keep getting a 'password incorrect' error even though I know it is right."
    print(f"\nProcessing Query: '{query}'")
    
    if client is not None:
        output = run_prompt_chain(query)
        
        print("\n Final Prompt Chain Output (5 Stages):")
        
        # Display the results clearly
        stage_names = [
            "1. Intent Interpretation", 
            "2. Possible Categories", 
            "3. Final Category", 
            "4. Extracted Details", 
            "5. Final Response"
        ]
        
        for i, step_result in enumerate(output):
            print(f"  {stage_names[i]}:")
            print(f"    -> {step_result}")
    else:
        print("\nSkipping execution due to Gemini client initialization error.")