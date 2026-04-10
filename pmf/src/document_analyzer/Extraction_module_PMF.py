import io
import os
import re
import shutil 
import logging
from docx import Document
import json
import pandas as pd
from io import StringIO
import streamlit as st
from datetime import datetime
from docxcompose.composer import Composer
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

from src.document_generate.dynamic_template_PMF import handle_user_message
from src.document_ingestion.data_collection import data_extraction,convert_all_doc_to_docx_in_folder,extract_text_from_xlsx
from src.document_analyzer.json_converter import extract_text_from_word
from src.document_generate.Assembling_appendix_PMF import assemble_appendices,convert_word_to_pdf
from src.document_analyzer.contents import refresh_toc_with_word,extract_headings_with_tables
from src.document_retriever.Vector_db import DocumentRetriever
from src.eval.eval_config import get_eval_rules
from src.eval.eval_utils import evaluate_run
from src.eval.eval_store import save_eval_run



# Function to split the template text into two parts based on the section header
def Template_to_list(text):
    if text is None:
        return [], []     # Handle the case where text is None by returning empty lists
    
    sections = [section.strip() for section in text.split('$') if section.strip()]
    split_index = None
    for i, item in enumerate(sections):
        if "DEVICE DESCRIPTION & PRODUCT SPECIFICATION".lower() in item.lower():
            split_index = i
            break

    if split_index is not None:
        part1 = sections[:split_index]
        part2 = sections[split_index:]
    else:
        part1 = sections
        part2 = []

    return part1, part2


# Convert the list of strings into a dictionary with unique keys
def convert_dict(list):

    result_dict = {}
    key_count = {}

    for item in list:
    # Split into key and value at first newline
        parts = item.split('\n', 1)
        if len(parts) == 2:
            key, value = parts[0].strip(), parts[1].strip()
        else:
            # If no newline found, use whole as key and empty value
            key, value = parts[0].strip(), ""

        # Handle duplicate keys by counting
        if key in key_count:
            key_count[key] += 1
            new_key = f"{key}_{key_count[key]}"
        else:
            key_count[key] = 1
            new_key = key

        result_dict[new_key] = value
    return result_dict



def extraction_pmf(template_file_path):

    load_dotenv()
    AZURE_KEY = os.getenv('AZURE_KEY', 'dcda59d482da43c9b43a07674e41fe89')  # Replace with your actual key or keep it in .env
    AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT', 'https://api.geneai.thermofisher.com/dev/gpt4o')
    AZURE_NAME = os.getenv('AZURE_NAME', 'gpt-4o')
    AZURE_VERSION = os.getenv('AZURE_VERSION', '2024-05-01-preview')

    llm = None
    client = None
    try:
        llm = AzureChatOpenAI(
            deployment_name=AZURE_NAME,
            openai_api_key=AZURE_KEY,
            openai_api_base=AZURE_ENDPOINT,
            openai_api_version=AZURE_VERSION,
            openai_api_type="azure",
            temperature=0.1,
        )

        client = AzureOpenAI(
        api_key=AZURE_KEY,  
        api_version=AZURE_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
)
    except Exception as e:
        print("Failed to initialize AzureChatOpenAI!")
        print("Error:", e)


                
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_artifacts = {
        "timestamp": timestamp,
        "template_file": template_file_path,
        "site_name": st.session_state.get("SiteName", ""),
        "model_name": AZURE_NAME,
        "sections": [],
    }
    

 
   
    # ------------------------------------- Devide json into 2 Part--------------------------------------------------------------------------------------------
    
    template_content = extract_text_from_word(template_file_path)
    part1,part2=Template_to_list(template_content)
    template_json=convert_dict(part1)
    
                    
#--------------------------------------------------------------------------------------------------------------  
    
    # Output doc with footer header
    template_path_1 = fr"templates\\output_template_1.docx"
    template_path = fr"templates\\output_template_PMF.docx"
    # Create a new file name
    new_file_name = fr"PMF_Output_{timestamp}.docx"
    new_file_name_pdf = fr"PMF_Output_{timestamp}.pdf"
    
    
    
    first_doc = Document(template_path)
    for paragraph in first_doc.paragraphs:
        for run in paragraph.runs:
            if "[Site Name]" in run.text:
                # Replace the text
                run.text = run.text.replace("[Site Name]", st.session_state.SiteName)
                # Make the replaced text bold
                run.bold = True
    
    
    doc = Document(template_path_1)

   


    
    

# --------------------------------Iterate over after executive Summary--------------------------------------------------------------------
    doc_1=doc
    
    base_folder = fr"data/artifacts/Extracted_folder"
    folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    st.write(folders)
    if len(folders)<1:
        base_folder = rf"data/artifacts/Extracted_folder"
    else:
        base_folder = rf"data/artifacts/Extracted_folder/{folders[0]}"


    # convert all doc files to docx in the folder
    convert_all_doc_to_docx_in_folder(base_folder) 

    # Initialize the DocumentRetriever with the base folder and store the vector database in the specified path
    retriever = DocumentRetriever(base_folder)
    retriever.process_documents()
    
    st.write("Extracting text from documents...")


    # Check if the uploaded file is a CSV or Excel file and read it into a DataFrame
    data_as_string = ""
    if st.session_state.uploaded_excel is not None:
   
        data_as_string = extract_text_from_xlsx(st.session_state.uploaded_excel)

        st.write(data_as_string)
    



    for key,value in template_json.items():
       
        value = value.replace("[Site Name]", st.session_state.SiteName)
        value_ls= value.split('@!')
        
        flag=1
        index=1
        
        st.write(key)
        first_25 = value[:80]
        st.write(first_25)
        input_file_path=''
        st.write("")
        st.write("")

        section_json={key:value}
        try:    

            input_file_text=""
            paths = []
            # Retrive the documents name from the vector database
            if len(value_ls)>1:
                st.write(value_ls[1])
                results = retriever.search(value_ls[1], top_k=5)

                for i, (path, filename, score) in enumerate(results):
                    paths.append(path)
                st.write("Retrieved documents:")
                st.write(paths)
                    
                input_file_text +="excel data:\n" + data_as_string + "\n\n"+ "########################################\n\n"
                input_file_text += data_extraction(paths)

                st.write("-----------------------------------------------------------------------------------------------------------------------------")
              
            elif "static" not in key.lower(): 
                base_folder_ls=[base_folder]

                input_file_text +="excel data:\n" + data_as_string + "\n\n"+ "########################################\n\n"
                input_file_text += data_extraction(base_folder_ls)
            
            

            # try:   
            if "static" not in key.lower(): 


                    response_data = handle_user_message(llm,client,key,value_ls[0],doc_1,flag,index,input_file_text,input_file_path)
                
            else:
                # pass
                response_data = handle_user_message(llm,client,key,value_ls[0],doc_1,flag,index)  

            run_artifacts["sections"].append(
                {
                    "section_key": key,
                    "prompt_text": value_ls[0] if len(value_ls) > 0 else "",
                    "retrieval_query": value_ls[1] if len(value_ls) > 1 else "",
                    "retrieved_paths": paths,
                    "input_text_size": len(input_file_text or ""),
                    "is_static": "static" in key.lower(),
                    "agent_result": response_data,
                    "generated_text": (response_data or {}).get("result", {}).get("generated_text", ""),
                }
            )
        except Exception as e:
            st.write("Error in processing the section:")
            st.write(f"An error occurred: {e}")
            continue
                
           
            st.write("----------------------------------------------------------------------------------------------------------------------------")    

         

   

    # Save the document to a new file----------------------------------------------------------------------------------------------------
    new_file_path = fr"data/artifacts/generated output file\\{new_file_name}" 
    new_file_path_temp =rf"data/artifacts/generated output file\Temp_{new_file_name}"
    final_document_doc = rf"data/artifacts/generated output file\\Final_{new_file_name}"
    new_file_path_pdf =rf"data/artifacts/generated output file\\{new_file_name_pdf}"
    final_document = fr"data/artifacts/generated output file\\Final_{new_file_name_pdf}" 

    

    # Generated file save in doc
    composer = Composer(doc_1)
    composer.save(new_file_path_temp)


    # add Content to the document
    doc_t=extract_headings_with_tables(new_file_path_temp,0, final_document_doc)
    file_path_abs = os.path.abspath(final_document_doc) 
    refresh_toc_with_word(file_path_abs)

    
    # Add cover page to the document
    doc1 = Document(final_document_doc)
    composer1 = Composer(first_doc)
    composer1.append(doc1)
    composer1.save(new_file_path_temp)
    file_path_abs = os.path.abspath(new_file_path_temp)

    # create a link to the document
    link_doc = Document(new_file_path_temp)
    output = io.BytesIO()
    link_doc.save(output)
    output.seek(0)  

    run_artifacts["final_doc_path"] = new_file_path_temp
    rules = get_eval_rules()
    evaluation = evaluate_run(run_artifacts, rules)
    eval_file = save_eval_run(run_artifacts, evaluation)
    st.session_state["last_eval_file"] = eval_file
    st.session_state["last_eval_score"] = evaluation.get("document_scores", {}).get("overall_score")
    

   



  


# #------------------------Appendices assembling---------------------------------------------------------
    
    # # Define the base folder path
    # base_folder = r"data/artifacts/Extracted_folder"
    # folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    # if len(folders)<1:
    #     base_folder = rf"data/artifacts/Extracted_folder"
    # else:
    #     base_folder = rf"data/artifacts/Extracted_folder/{folders[0]}"

    # # file_path_abs=r"C:\Users\mihir.vasoya\OneDrive - Thermo Fisher Scientific\Desktop\Git-Integration\RegulatoryDocGen\generated output file\PMF_German_9_7_25.docx"
    # convert_word_to_pdf(file_path_abs,new_file_path_pdf)
    
    

    # res=assemble_appendices(base_folder, new_file_path_pdf, final_document)
    # print(res)









    # final_document=os.path.abspath(final_document) 
   

    # output=None
    return output,final_document,new_file_name_pdf

    


