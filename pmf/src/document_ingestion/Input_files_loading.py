import os
import re
from fuzzywuzzy import process
import streamlit as st
import win32com.client as win32
import pythoncom



 
DATA_PATH_MAPPING_EXECUTIVE_SUMMARY = {
    "executive summary": [],
    "sterilization related details": [],
    "regulatory status of the similar device": [],
    "risk management":[],
    "marketing history, introduction date, device models, summarize marketing history, device introduction": [],
    "marketing history table, sales quantity, STP ULT, global sales, model number, units sold, marketing data": [],
    "domestic price": [],
    "safety and performance related information": []}
 

EXECUTIVE_SUMMARY_SYNONYMS = {
    "executive summary": [],
    "sterilization related details": [],
    "regulatory status of the similar device": [],
    "risk management":[],
    "marketing history, introduction date, device models, summarize marketing history, device introduction": ['Sales Data','ISO 13485'],
    "marketing history table, sales quantity, STP ULT, global sales, model number, units sold, marketing data":['Sales Data',"TDS","Technical Datasheet","Technical Data Sheet"],
    "domestic price": ['Sales data',"TDS","Technical Datasheet","Technical Data Sheet"],
    "safety and performance related information": ['Sales Data', 'ISO 13485']}


# Define a mapping of categories to their data paths

DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING = {
    "device description": [],
    "device information": [],
    "materials of construction": [],
    "intended use": [],
    "indications": [],
    "instructions for use": [],
    "contraindications": [],
    "warnings precautions": [],
    "potential adverse effects": [],
    "intended patient population": [],
    "medical condition": [],
    "principle of operation": [],
    "working principle": [],

    "novel features": [],
    "accessories and other products that are intended to be used in combination": [],
    "optional accessories": [],
    "various configurations or variants of the device": [],
    "key components": [],
    "material information of key functional elements": [],
    "physical chemical and microbiological characterization": [],
    "ionizing radiation": [],
    "product specification": [],
    "performance attributes": [],
    "device accessories": [],
    "optional parts delivered with the device": [],
    "manufacturing processes": [],
    "risk analysis and control summary": [],
    "electrical safety and electromagnetic capability testing": [],
    "performance evaluation intended use": [],
    "performance evaluation test details": [],
    "biocompatibility": [],
    "medicinal substances": [],
    "biological safety": [],
    "sterilization": [],
    "animal studies": [],
    "stability data": [],
    "post marketing surveillance data": [],
}

FROM_DEVICE_DESCRIPTION_CATEGORY_SYNONYMS = {
    "device description": ["User Manual"],
    "device information": ["TDS","Technical Datasheet","Technical Data Sheet"],
    "materials of construction": ["Safety Report"],
    "intended use": ["User Manual"],
    "indications": ["User Manual"],
    "instructions for use": ["User Manual"],
    "contraindications": ["User Manual"],
    "warnings precautions": ["User Manual"],
    "potential adverse effects": [],
    "intended patient population": ["User Manual"],
    "medical condition": ["User Manual"],
    "principle of operation": [],
    "working principle": ["image"],

    "novel features": [],
    "accessories and other products that are intended to be used in combination": ["TDS","Technical Datasheet","Technical Data Sheet","User Manual","Brochure"],
    "optional accessories": ["User Manual"],
    "various configurations or variants of the device": ["TDS","Technical Datasheet","Technical Data Sheet"],
    "key components": ["Safety Report"],
    "material information of key functional elements": [],
    "physical chemical and microbiological characterization": ["Safety Report"],
    "ionizing radiation": ["EMC reports"],
    "product specification": ["TDS","Technical Datasheet","Technical Data Sheet"],
    "performance attributes": ["TDS","Technical Datasheet","Technical Data Sheet"],
    "device accessories": ["User Manual"],
    "optional parts delivered with the device": ["User Manual"],
    "manufacturing processes": [],
    "risk analysis and control summary": ["Risk Management Files"],
    "electrical safety and electromagnetic capability testing": ["EMC reports","Safety Report"],
    "performance evaluation intended use": ["User Manual"],
    "performance evaluation test details": ["V&V","Verification and Validation Reports"],
    "biocompatibility": ["V&V","Verification and Validation Reports"],
    "medicinal substances": ["V&V","Verification and Validation Reports"],
    "biological safety": ["V&V","Verification and Validation Reports"],
    "sterilization": ["V&V","Verification and Validation Reports"],
    "animal studies": ["V&V","Verification and Validation Reports"],
    "stability data": ["V&V","Verification and Validation Reports"],
    "post marketing surveillance data": ["Sales Data", "Complaint, AE,FSCA","Complaints, Adverse events, FSCA", "ISO 13485"]
}

def get_available_folders(base_folder):
    """Retrieve the list of available folders in the base folder."""
    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"Base folder '{base_folder}' not found.")
    return os.listdir(base_folder)


def map_folders_from_device_description_data_paths(available_folders, base_folder):
    """Map folder names to their corresponding data paths and update DATA_PATH_* variables."""
    
    global DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING, FROM_DEVICE_DESCRIPTION_CATEGORY_SYNONYMS


    for category in DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING.keys():
        DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING[category] = []  # Reset the list for each category
    
    for folder in available_folders:
        # folder_normalized = normalize_text(folder)
        folder_normalized = folder.lower()
        # print('folder_normalized...', folder_normalized)
        if folder_normalized.startswith("software"):
            # print(f'Skipping folder "{folder}" as it starts with "software"')
            continue  # Skip this folder entirely

        if folder_normalized.startswith("sop") and (folder_normalized.endswith(".pdf") or folder_normalized.endswith(".docx") or folder_normalized.endswith(".doc")):
            
            sop_file_path = os.path.join(base_folder, folder)

            if folder_normalized.endswith(".doc"):
                sop_file_path_abs = os.path.abspath(sop_file_path).replace("/", "\\")

                pythoncom.CoInitialize()
                try:
                    word = win32.gencache.EnsureDispatch('Word.Application')
                    doc = word.Documents.Open(sop_file_path_abs)
                    
                    output_path = sop_file_path_abs.replace('.doc', 'abc.docx')
                    doc.SaveAs(output_path, FileFormat=16)
                    doc.Close()
                    word.Quit()
                finally:
                    pythoncom.CoUninitialize()
                sop_file_path = sop_file_path.replace('.doc', '.docx')
            output_path = sop_file_path.replace("\\" , "/")

            # Normalize path for cross-platform compatibility
            if sop_file_path not in DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING["manufacturing processes"]:
                DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING["manufacturing processes"].append(output_path)
                continue  # Skip further processing for "manufacturing processes"

        for category, synonyms in FROM_DEVICE_DESCRIPTION_CATEGORY_SYNONYMS.items():
            folder_path=r""
            # if category == "manufacturing processes":
            #     handle_sop_files(folder_normalized)
            #     continue  # Skip SOP files for other categories
            # normalized_synonyms = [normalize_text(synonym) for synonym in synonyms]
            normalized_synonyms = [synonym.lower() for synonym in synonyms]
            # print(f'Checking folder "{folder}" against category "{category}"')
            # print('folder_normalized...', folder_normalized)
            # print('normalized_synonyms...', normalized_synonyms)
            if folder_normalized == "sales data" and category == "device information":
                # print(f'Skipping folder "{folder}" for category "{category}"')
                continue

            if folder_normalized in normalized_synonyms:
                # print('Matched folder (direct):', folder)
                if folder not in DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING[category]:
                    folder_path=rf"{base_folder}/{folder}"
                    DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING[category].append(folder_path)
            else:
                best_match = process.extractOne(folder_normalized, normalized_synonyms)
                # print('best_match...', best_match)
                if best_match and best_match[1] >= 90:
                    print(f'Fuzzy matched folder "{folder}" to category "{category}" with score {best_match[1]}')
                    if folder not in DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING[category]:
                        folder_path=rf"{base_folder}/{folder}"
                        DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING[category].append(folder_path)
    return {"Status": "Success", "Result": "Folders from Device Description mapped Successfully"}

# # Map available folders to data paths
# res=map_folders_from_device_description_data_paths(available_folders)
# # validate_mapped_folders()

# print(DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING)


def map_executive_summary_folders(available_folders,base_folder):
    """Map folder names to their corresponding executive summary data paths."""
    
    global DATA_PATH_MAPPING_EXECUTIVE_SUMMARY, EXECUTIVE_SUMMARY_SYNONYMS

    # Reset the list for each executive summary category
    for category in DATA_PATH_MAPPING_EXECUTIVE_SUMMARY.keys():
        # st.write(category)
        DATA_PATH_MAPPING_EXECUTIVE_SUMMARY[category] = []
        # st.write(DATA_PATH_MAPPING_EXECUTIVE_SUMMARY)

    # Iterate through available folders
    for folder in available_folders:
        folder_normalized = folder.lower()

        # Exclude folders starting with "software"
        if folder_normalized.startswith("software"):
            
            # print(f'Skipping folder "{folder}" as it starts with "software"')
            continue  # Skip this folder entirely

        # Process executive summary categories
        for category, synonyms in EXECUTIVE_SUMMARY_SYNONYMS.items():
            folder_path=r""
            normalized_synonyms = [synonym.lower() for synonym in synonyms]
            # st.write(folder_normalized)
            # st.write(normalized_synonyms)
            # print(f'Checking folder "{folder}" against executive summary category "{category}"')
            # print('folder_normalized...', folder_normalized)
            # print('normalized_synonyms...', normalized_synonyms)

            # Direct match
            # if folder_normalized in normalized_synonyms:
            #     print(f'Matched folder (direct): {folder}')
            #     if folder not in DATA_PATH_MAPPING_EXECUTIVE_SUMMARY[category]:
            #         DATA_PATH_MAPPING_EXECUTIVE_SUMMARY[category].append(folder)
            #     break
            if folder_normalized in normalized_synonyms:
                # st.write("normalized_synonyms...")
                # st.write(folder_normalized)
                # st.write(normalized_synonyms)
                # print('Matched folder (direct):', folder)
                if folder not in DATA_PATH_MAPPING_EXECUTIVE_SUMMARY[category]:
                    folder_path=rf"{base_folder}/{folder}"
                    DATA_PATH_MAPPING_EXECUTIVE_SUMMARY[category].append(folder_path)

            else:
                # st.write("fuzzy match...")
                # st.write(folder_normalized)
                # st.write(normalized_synonyms)
                best_match = process.extractOne(folder_normalized, normalized_synonyms)
                # print('best_match...', best_match)
                if best_match and best_match[1] >= 90:
                    print(f'Fuzzy matched folder "{folder}" to category "{category}" with score {best_match[1]}')
                    if folder not in DATA_PATH_MAPPING_EXECUTIVE_SUMMARY[category]:
                        folder_path=rf"{base_folder}/{folder}"
                        DATA_PATH_MAPPING_EXECUTIVE_SUMMARY[category].append(folder_path)

    return {"Status": "Success", "Result": "Executive Summary related Folders mapped Successfully"}

#############################

# from Input_files_loading import (
#     get_available_folders,
#     map_folders_from_device_description_data_paths,
#     map_executive_summary_folders,
#     DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING,
#     DATA_PATH_MAPPING_EXECUTIVE_SUMMARY
# )

# # Define base folder
# base_folder = "doc_retrieval/STP ULT"

# # Get available folders
# available_folders = get_available_folders(base_folder)

# # Map folders

# res=map_folders_from_device_description_data_paths(available_folders, base_folder)
# print("res...",res)
# print("device_desc...",DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING)

# res1=map_executive_summary_folders(available_folders)
# print("res1...",res1)
# print("exce summary...",DATA_PATH_MAPPING_EXECUTIVE_SUMMARY)