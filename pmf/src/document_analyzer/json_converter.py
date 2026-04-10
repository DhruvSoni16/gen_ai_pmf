import re
import openai
import time
import json
import os
from dotenv import load_dotenv
from docx import Document
import streamlit as st

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY', '')

def extract_text_from_word(docx_file,doc=""):
    """
    Extracts text from a Word document without any additional cleaning or processing.
    """
    try:
        if doc=="":
            doc = Document(docx_file)
            
        else :
            doc=doc

        result = ""
        for block in doc.element.body:
            if block.tag.endswith('p'):
                # Handle paragraphs manually (like in your original logic)
                paragraph = block
                text = ''.join(node.text for node in paragraph.xpath('.//w:t') if node.text)
                if text.strip():
                    result += text.strip() + "\n"

            elif block.tag.endswith('tbl'):
                # Convert XML table to docx.table.Table
                for tbl in doc.tables:
                    if tbl._element == block:
                        table_data = []
                        for row in tbl.rows:
                            row_data = {}
                            for i, cell in enumerate(row.cells):
                                row_data[f"col_{i}"] = cell.text.strip()
                            table_data.append(row_data)

                        table_json = json.dumps(table_data, ensure_ascii=False)
                        result += table_json + "\n"
                        break  # Only match once to avoid duplicates

        return result.strip()

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def key_stucture(json_text):
    merged_data = {}
    previous_key=""
    current_key = None
    current_value = ""
    for key, value in json_text.items():
        # Extract base key by removing digits and hyphens
        base_key = re.sub(r'-\d+', '', key)
        
        if current_key is None:  # Start with the first key
            
            previous_key=key
            current_key = base_key
            current_value = value
            
        elif base_key == current_key:  # Continue merging if the key matches
            current_value += f"\n\n{value}"
        else:  # Different key encountered, save the current group and start a new one
            merged_data[previous_key] = current_value
            current_key = base_key
            current_value = str(value)
            previous_key=key

    # Add the last merged group to the dictionary
    if current_key:
        merged_data[key] = current_value
        
    return merged_data

def json_conveter(pdf_content,llm,client):
  
    
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {  
                "role": "system",
                "content": """
                You are a highly accurate and rule-bound assistant. Your task is to convert the input into a **strictly valid JSON format** without omitting or altering any content.

                - The document will be structured under **five main keys**: `Text`, `Table`, `Image`, `Static_Text` or `Static_Table`, and `Web Scraping` if present.
                - **Do not create sub-keys** under any of the main keys. Store the content exactly **as-is** under the appropriate key.
                - **Preserve all heading numbers (e.g., 1, 2, 3...)** exactly as they appear.
                - **Do not skip or rephrase any 'Instructions' or 'Notes'**—they must be included in full in the corresponding key's value.
                - The goal is a one-to-one mapping from the source content to the JSON output. Absolutely no restructuring, rewording, or filtering is allowed.
                - Ensure that the JSON is **valid and well-formed**. Use double quotes for keys and string values, and ensure proper escaping of special characters.
                - do not omit any text, labels, or special formatting in the output."""
            },

            {   
                "role": "user",
                "content": f"""
                You are a helpful assistant that outputs **only JSON** without any additional text or comments outside the JSON.and And do not ommiting anything and adhering to the following rules:
                    
                    1. **Preserve All Headings with Numbers**: Every heading in the input document must appear exactly as-is, including its number. Do not modify or omit the numbers in the headings.

                    2. **Category Differentiation**:
                    - The document is divided into Five main categories(keys): `Text`, `Table`, and `Image`,`Static_Text or Static_Table `,Web Scrapping.
                    - If any category appears multiple times in the input, append a unique identifier (e.g., `Text-1`, `Text-2`, `Table-1`, etc.) to differentiate them.
                    - Use the main category names (`Text`, `Table`, `Image`, `Static`,'Web Scrapping') as top-level keys.
                    - **don't make sub key of top-level key write as-is in key value**
            
                    3. **Retain Original Formatting**:
                    - For sections labeled as `Text`, retain the input content **as-is do not ommiting any text**, including any numbers, labels, or special formatting.
                    - **For sections labeled as `Table` , use the given headings and structure provided in the input. **please Do not add placeholders like "user input required" beside column names and do not ommiting any this just write "as-is" under the table key****.
                    - - For sections labeled as `Static`, retain the input content **as-is do not ommiting any text**, including any numbers, labels, or special formatting.
                    
                    4. **Handle Numbers with Labels**:
                    - If the document contains numbers with labels (e.g., "1 Introduction", "Table 3 Device Features"), retain them exactly "as-is" they are in the output. Do not alter or remove these numbers.
                    **Don't add "user input required" in table section infront of columns list **
                   
                        
                    Here is the document content:

                    DOC Text: {pdf_content}

                    Generate the structured JSON output strictly adhering to the above instructions . and do not ommiting any text, labels, from the input document. The output should be a valid JSON object with the following structure:


                """
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    
   
    
    # Parse the JSON response from the LLM
    response_dict = response.model_dump()
    response_text = response_dict['choices'][0]['message']['content'].strip()
    # st.write("Response Text:", response_text)
    json_output = json.loads(response_text)
    # json_output1=key_stucture(json_output)
    return json_output

