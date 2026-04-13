
import openai
import json
import pytesseract
import logging
import pandas as pd
import time
from io import StringIO
import numpy as np
import os
import streamlit as st
import copy
from copy import deepcopy
from docxcompose.composer import Composer
from docx import Document

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_analyzer.text import process_text_with_GPT
from src.document_analyzer.table import derived_table,derived_static_table
from src.document_analyzer.image import extract_images_with_fallback,image_selection_1
from src.document_generate.doc_generate import save_text_in_document_1
from src.scraping.scrap_2 import scrapping

pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'
input_tokens=0,


# Load the config file
with open('Config/configuration.json', 'r') as f:
    config = json.load(f)


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY', '')
AZURE_KEY = os.getenv('AZURE_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT', '')
AZURE_NAME = os.getenv('AZURE_NAME', '')
AZURE_VERSION = os.getenv('AZURE_VERSION', '')

llm = None
try:
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_NAME,     
        openai_api_key=AZURE_KEY,
        azure_endpoint=AZURE_ENDPOINT,   
        openai_api_version=AZURE_VERSION,
        temperature=0.1,
        # openai_api_type="azure" is no longer needed, the new package infers it!
    )
except Exception as e:
    print("Failed to initialize AzureChatOpenAI!")
    print("Error:", e)






logging.basicConfig(
    filename='logs/app.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


main_prompt = """You are a Triaging Agent. Your role is to evaluate the user's query key  and **route it to the relevant only and only one agent do not route more then 1 agent** . 
  if key is text related then Text Extraction Agent call,
  if key is Image related then Image Extraction Agent,
  if key is Table related then Table Extraction Agent,
  if key is Web Scrapping related then Web Extraction Agent,
  if key is Static related like static_table,static_text simmilar, Then Static Extraction Agent 

- Text Extraction Agent: Summary generate, Normal text generate, Extract warning and precaution(Caution). (if do not mention any table or iamge related by default call text agent) 
- Table Extraction Agent: Derived table and Normal table
- Image Extraction Agent: Extract images
- Web Extraction Agent:   Extract web part
- Static Extraction Agent: Extract Static part

Use the send_query_to_agents tool to forward the **user's query to the relevant only any only one agents**."""

text_prompt = """You are a Text Extraction Agent. Your role is to evaluate the user's query key and route the only and only one tool. Normal text generate, Extract warning or precaution using the following tools:

  if key is normal text related then Normal_text_generate tool call,
  if key is warning precaution related then Extract_warning_and_precaution tool call


- Normal_text_generate if do not mention warning precaution then consider as Normal_text_generate
- Extract_warning_and_precaution :

Note: **route it to the relevant only and only one tool please do not route more then 1 tool at time** 
"""


Table_prompt = """You are a Table Extraction Agent. Your role is to by default Extract_table function call for extract table from the text:
- Extract_table
"""

Iamge_prompt = """You are an Image Extraction Agent. Your role is to  Extract Image by default using the following tool:
- Image_Extraction
"""

triage_tools = [
    {
        "type": "function",
        "function": {
            "name": "send_query_to_agents",
            "description": "Sends the user query to relevant only and only one agent based on their capabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "An array of agent names to send the query to."
                    },
                    "query": {
                        "type": "string",
                        "description": "The user query to send."
                    }
                },
                "required": ["agents", "query"]
            }
        },
        "strict": True
    }
]


Text_Extraction_tools = [
    {
        "type": "function",
        "function": {
            "name": "Normal_text_generate",
            "description": "if summary and warning precaution do not mention in user query then consider as Normal text.extract the text for the given section of the text",
            
        },
        "strict": True

    }
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "Extract_warning_and_precaution",
    #         "description": "if warning and precausion  related text mention in text then call Extract_warning_and_precaution. Extract the warning and precaution fron text",
           
    #     },
    #     "strict": True
    # }
]

Table_Extraction_tools = [
    {
        "type": "function",
        "function": {
            "name": "Extract_table",
            "description": "by default Extract_table function call extract table from the text ",
           
        },
        "strict": True
    },
]

Image_Extraction_tools = [
    {
        "type": "function",
        "function": {
            "name": "Image_Extraction",
            "description": "extract Image from the text ",
           
        },
        "strict": True
    },
]






def handle_text_agent(llm,client,key, value, doc,flag,index,extract_text,pdf_bytes):
    
    
    start_time = time.time()
    flag = 0
    user_query = "Key=" + key + ":" + value
    messages = [{"role": "system", "content": text_prompt}]
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model=AZURE_NAME,
        messages=messages,
        temperature=0.0,
        tools=Text_Extraction_tools  # specify the function call
    )
    response_time = time.time() - start_time
    response_dict = response.model_dump()
    

    tool_call = response_dict["choices"][0]["message"]["tool_calls"][0]
    function_name = tool_call["function"]["name"]
    print(function_name)
    
    generated_text = ""

    if tool_call["function"]["name"] == "Normal_text_generate":

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=170000, chunk_overlap=1000)
        text_chunks = text_splitter.create_documents([extract_text])
        st.write("chunk size",len(text_chunks))

        process_text=""
        for i in text_chunks:
            
            process_text+=process_text_with_GPT(i,value,llm,client)

        refinement_instruction = """Ensure the extracted information is unique, non-repetitive, and provides a clear overview adhering to the original extraction goals.

        - Preserve the EXACT formatting as below:
        • Replace all numbering (e.g., 6.1, 6.1.1) with bullet points using "•" symbols.
        • Maintain paragraph breaks and spacing.
        • Do not add bold text to headings.
        • Ensure No subheadings are used in the final response."""
        
        combined_instruction = f"""
            Instructions:
            {value}
        
            Refinement Instructions:
            {refinement_instruction}
        """


        # Refine the text to remove repetitions and ensure uniqueness
        refined_text = process_text_with_GPT(process_text, combined_instruction,llm,client)

        st.write("refined_text",process_text)

        split_string = refined_text.split('#!')
        split_string = [s.strip() for s in split_string if s.strip()]

        for i in split_string:
            
            if i.lower().startswith('table') or i.lower().startswith('table') or i.lower().endswith('}') or "{[" in i:

                flag=2
                new_table=derived_static_table(i,llm,client)
                save_text_in_document_1(new_table,doc,flag,value,index)                    
            else:
                    flag=0
                    save_text_in_document_1(i, doc, flag,index)
                    generated_text += ("\n" + i if generated_text else i)
                        
            
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=175000, chunk_overlap=2000)
        text_chunks = text_splitter.create_documents([extract_text])
        # text_chunks = text_splitter.create_documents([all_text])

        process_text=""
        for i in text_chunks:
            process_text+=process_text_with_GPT(i,value,llm,client)
        

        if len(text_chunks) > 1:
            # st.write("1")
            refinement_instruction = f"""
            -Do not mention the PDF texts, documents, or the source of the information provided.
            -The Input Text consists of chunk-wise extracted data. Your task is to analyze **all chunks collectively** and generate the final output based on the full context.
            -Strictly follow the formatting and content rules defined in the instruction below.
            Must be in the following format:
            {value}

            """
            # Refine the text to remove repetitions and ensure uniqueness
            refined_text = process_text_with_GPT(process_text, refinement_instruction,llm,client)

        else:
            refined_text = process_text

        save_text_in_document_1(refined_text, doc, flag,index)
        generated_text = refined_text

    return {
        "agent": "Text Extraction Agent",
        "tool": function_name,
        "generated_text": generated_text,
        "response_time_sec": response_time
    }
    
    
def handle_table_agent(llm,client,key,query, doc,flag,index,extract_text):
    flag = 2

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=175000, chunk_overlap=2000)
    text_chunks = text_splitter.create_documents([extract_text])

    refined_text = {}
    try:
    # Collect the dictionaries from each chunk
        table_data_list = []
        for i in text_chunks:
            table_data_list.append(derived_table(i.page_content, query,llm,client))  # Access page_content

        # Combine the dictionaries into a single dictionary
        combined_table_data = {}
        for data in table_data_list:
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in combined_table_data:
                        if isinstance(value, list):
                            combined_table_data[key].extend(value)
                        else:
                            logging.warning(f"Value for key '{key}' is not a list, skipping concatenation.")
                    else:
                        combined_table_data[key] = value
            else:
                logging.warning(f"Skipping non-dictionary data: {data}")

        if len(text_chunks) > 1:
            # refinement_instruction = "Ensure the extracted information is unique, non-repetitive, and provides a clear overview adhering to the original extraction goals."
                
            
            refinement_instruction = """Ensure the extracted information is unique, non-repetitive, and provides a clear overview adhering to the original extraction goals. Consolidate all the extracted information into a single, coherent Table that includes all relevant details without duplication."""
                
            combined_instruction = f"""
                Original Extraction Goals:
                {query}

                Refinement Instructions:
                {refinement_instruction}
            """

            # Convert the combined_table_data to a string representation for refinement
            combined_table_string = str(combined_table_data)
            refined_text = derived_table(combined_table_string, combined_instruction,llm,client)

        
        else:
            refined_text = combined_table_data
            

    except Exception as e:
        print(f"Error during processing: {e}")

        if isinstance(refined_text, dict):  # Check if table is a dictionary
            save_text_in_document_1(refined_text, doc, flag, index)
        else:
            print("Table is not a dictionary. Please check the output of `derived_table_cer_1()`.")

    return {
        "agent": "Table Extraction Agent",
        "tool": "Extract_table",
        "generated_text": str(refined_text) if refined_text else "",
        "response_time_sec": 0
    }
        

def handle_image_agent(query, doc, extract_text,pdf_bytes):
    
        flag=1
        index=0
        split_text = query.split(":")
        st.write(split_text[1])
        try:
    
            pdf_path=rf"{pdf_bytes[0]}/Image.pdf"
            st.write(pdf_path)

            image_save = extract_images_with_fallback(pdf_path,r"data\artifacts\ExtractedImages2",split_text[1],flag)


            image_selection,image_token=image_selection_1(r"data\artifacts\ExtractedImages2","Refrigeration System")

            save_text_in_document_1(image_selection,doc,flag,index)
        except:
             save_text_in_document_1(None,doc,flag,index)

        return {
            "agent": "Image Extraction Agent",
            "tool": "Image_Extraction",
            "generated_text": "",
            "response_time_sec": 0
        }

 
def handle_static_agent(llm,client, key,value,doc,flag,index):
    
    if "text" in key.lower() :
        flag=0
        save_text_in_document_1(value, doc, flag,index)
        generated_text = value
        
    else:
        flag=2
        new_table=derived_static_table(value,llm,client)
        save_text_in_document_1(new_table,doc,flag,value,index)
        generated_text = str(new_table)

    return {
        "agent": "Static Extraction Agent",
        "tool": "Static",
        "generated_text": generated_text,
        "response_time_sec": 0
    }

  
def handle_web_agent(llm,client,key,value,doc,flag):
    pass
   
    
    


def handle_user_message(llm,client,key,value, doc,flag=1,index=1, extract_text="", pdf_bytes=''):
    start_time = time.time()
    user_query = "Key=" + key + ":" + str(value)
    
    user_message = {}
    conversation_messages = []
    user_message = {"role": "user", "content": user_query}
    conversation_messages.append(user_message)

    messages = []
    messages = [{"role": "system", "content": main_prompt}]
    messages.extend(conversation_messages)

    response = client.chat.completions.create(
        model=AZURE_NAME,
        messages=messages,
        temperature=0.0,
        tools=triage_tools  # specify the function call
    )
    response_time = time.time() - start_time
    response_dict = response.model_dump()

   

   
    # Process the response
    tool_call = response_dict["choices"][0]["message"]["tool_calls"][0]

    agent_result = {
        "agent": "",
        "tool": "",
        "generated_text": "",
        "response_time_sec": response_time
    }

    if tool_call["function"]["name"] == 'send_query_to_agents':
        agents = json.loads(tool_call["function"]["arguments"])['agents']
        query = json.loads(tool_call["function"]["arguments"])['query']
        
        for agent in agents:
            print("###########################################")
            print(agent + "------------") 
            
            if agent == "Text Extraction Agent":
                    
                agent_result = handle_text_agent(llm,client,key, value, doc,flag,index, extract_text, pdf_bytes)

            elif agent == "Table Extraction Agent":
                st.write("Table part trigger")
                
                agent_result = handle_table_agent(llm,client,key,str(value), doc,flag,index ,extract_text)
               

            elif agent == "Image Extraction Agent":
                # pass
                agent_result = handle_image_agent(value, doc, extract_text, pdf_bytes)

            elif agent == "Web Extraction Agent":
                pass
                # st.write("Web part trigger")
                # web=handle_web_agent(llm,client,key,value,doc,flag)
                
            elif agent == "Static Extraction Agent":
                st.write("Static part trigger")
                agent_result = handle_static_agent(llm,client,key,value,doc,flag,index)
            else:
                pass
                


    return {
        "router_response_time_sec": response_time,
        "selected_agents": agents if 'agents' in locals() else [],
        "query": query if 'query' in locals() else user_query,
        "result": agent_result
    }
