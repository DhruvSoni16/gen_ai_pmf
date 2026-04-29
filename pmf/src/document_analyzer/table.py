import openai
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import logging
import logging
from datetime import datetime
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import AzureChatOpenAI

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import json



from langchain_core.output_parsers import JsonOutputParser
import tiktoken

with open('Config/configuration.json', 'r') as f:
    config = json.load(f)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY', '')
logging.basicConfig(
    filename='logs/app.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)




timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def derived_table(pdf_text, reference_text, llm,client):
      # Enable verbose mode for token logging
    prompt_template = """
    

        {{
            PDF Text: {pdf_text}
        Reference_Text:{reference_text}
            
        }}

    
    Your task is to process the provided text and extract values to generate a structured JSON format. Follow the instructions and guidelines below:

        Instructions:

        JSON Structure Requirements:
        Organize extracted data under separate table_x keys (e.g., table_1 etc.) generate only one and one table  .
        Each table should be represented as an array of JSON objects, where each object contains key-value pairs for the corresponding columns and their extracted values.
        
        Data Extraction Rules:
        Extract values for specified columns from the provided PDF text. If a column is missing from the text, populate it with "User Input Required".
        For columns related to classifications or class, explicitly set their values as "User Input Required".
        If multiple values are found for a single column, split them into separate JSON objects.

        Note that table_name write "as-is" whichever you find in a Instruction.Example: if instruction Table Name:- "Table followed digit(d): XYZ is there then write in json table_name: "Table digit(d): XYZ" in output json.               
        example:-
        {{
        "table_1": [
            {{
            "heading": "Heading for Table 1 ",
            "table_name": "Table Name",
            "columns": [{{"columns":values,}},]
            }}
        ],
        "table_2": [
            {{
            "heading": "Heading for Table 2 ",
            "table_name": "Table Name",
            "columns":  [{{"columns":values,}},]
            }}
        ]
        }}

         Example must follow based on "reference_tex" if only table in referanece text then generate only one table don't go with "pdf_text"
        - do not generate "columns" key more then one times for one table
        - if any heding store write or replace generic name then replace with generic name with this part                
        """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["pdf_text", "reference_text"])
    formatted_prompt = prompt.format(pdf_text=pdf_text, reference_text=reference_text)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=formatted_prompt)
    ]

    # Call the AzureChatOpenAI model
    response = llm.invoke(messages)

    # Parse the JSON output using JsonOutputParser
    json_parser = JsonOutputParser()

    # response.content is a string - parse it using JsonOutputParser
    parsed_output = json_parser.parse(response.content)

    return parsed_output


def derived_static_table(reference_text,llm,client):
    print("-------------------------------------------------------------------------")
    print("Table_agent_calles")
   
    prompt_template = """
    

        {{
        
        Reference_Text:{reference_text}
            
        }}

    
    Your task is to process the provided text and restructure the table values to generate a structured JSON format. Follow the instructions and guidelines below:
        -**make sure output is only English language do not generate in another language**
    
        -If table columns name are splite in 2 or 3 rows then merge them into one row and generate the json format.
        
        Instructions:

        JSON Structure Requirements:
       - Organize extracted data under separate table_x keys (e.g., table_1, table_2, etc.) if multiple tables are identified.
       - Each table should be represented as an array of JSON objects, where each object contains key-value pairs for the corresponding columns and their extracted values.
       - Make sure Table name write "as-is" whichever you find in a Instruction.Example: if instruction "Table Name:"Table digit: XYZ"" is there then write ""Table digit: XYZ"" in output json.
        Data Extraction Rules:
        given Reference_Text has table data your task is just convert into corosponding json format
        If multiple values are found for a single column, split them into separate JSON objects.
        - Make sure Output JSON is well-structured and follows the specified format exactrly mentioned below.               
        example:-
        {{
        "table_1": [
            {{
            "heading": "Heading for Table 1 ",
            "table_name": "Table Name",
            "columns": [{{"columns":values,}},]
            }}
        ],
        "table_2": [
            {{
            "heading": "Heading for Table 1 ",
            "table_name": "Table Name",
            "columns":  [{{"columns":values,}},]
            }}
        ]
        }}

         
        - do not generate "columns" key more then one times for one table               
        """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["reference_text"])
    formatted_prompt = prompt.format(reference_text=reference_text)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=formatted_prompt)
    ]

    # Call the AzureChatOpenAI model
    response = llm.invoke(messages)

    # Parse the JSON output using JsonOutputParser
    json_parser = JsonOutputParser()

    # response.content is a string - parse it using JsonOutputParser
    parsed_output = json_parser.parse(response.content)

    return parsed_output