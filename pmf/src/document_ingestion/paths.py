import difflib
import re
from collections import Counter
import os
from src.document_ingestion.Input_files_loading import (
    get_available_folders,
    map_folders_from_device_description_data_paths,
    map_executive_summary_folders,
    DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING,
    DATA_PATH_MAPPING_EXECUTIVE_SUMMARY
)
import streamlit as st



# Define base folder
def maping_folder():
    """Map folder names to their corresponding data paths."""
    base_folder = fr"data/artifacts/Extracted_folder"
    folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    if len(folders)>1:
        base_folder = rf"data/artifacts/Extracted_folder"
    else:
        base_folder = rf"data/artifacts/Extracted_folder/{folders[0]}"

    # Get available folders
    available_folders = get_available_folders(base_folder)

    # Map folders
    res=map_folders_from_device_description_data_paths(available_folders, base_folder)
    print("res...",res)
    print("device_desc...",DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING)
    st.write(DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING)

    res1=map_executive_summary_folders(available_folders,base_folder)
    print("res1...",res1)
    print("exce summary...",DATA_PATH_MAPPING_EXECUTIVE_SUMMARY)
    st.write(DATA_PATH_MAPPING_EXECUTIVE_SUMMARY)


def normalize_text(text):
    """Normalize text by converting to lowercase and removing special characters."""
    import re
    return re.sub(r'\W+', ' ', text.lower())
 


def map_categories_to_json(json_data):
    """
    Maps keys in json_data to their corresponding data paths and content.
 
    Parameters:
        json_data (dict): Input JSON data containing text content.
 
    Returns:
        list: A list of dictionaries with key name, data path, and content.
    """
    print("json_data map_categories_to_json...", json_data)
    mapped_data = []  # List to store the mappings
    used_keys = set()  # Track used categories
 
    # Prioritize specific categories like "biocompatibility"
    priority_categories = ["biocompatibility"]
 
    # Handle priority categories first
    for category in priority_categories:
        if category in DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING:
            data_paths = DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING[category]
            for key, content in json_data.items():
                if category not in used_keys and category in normalize_text(content):
                    mapped_data.append({
                        "key_name": key,
                        "data_path": data_paths,
                        "content": content,
                    })
                    used_keys.add(category)
                    break  # Stop searching once a match is found for this category
 
    # Handle the remaining categories
    for category, data_paths in DATA_PATH_FROM_DEVICE_DESCRIPTION_MAPPING.items():
        if category in priority_categories:
            continue  # Skip already handled priority categories
 
        for key, content in json_data.items():
            if category not in used_keys and category in normalize_text(content):
                mapped_data.append({
                    "key_name": key,
                    "data_path": data_paths,
                    "content": content,
                })
                used_keys.add(category)
                break  # Stop searching once a match is found for this category
 
    # Handle unmatched keys
    for key, content in json_data.items():
        if key not in [entry["key_name"] for entry in mapped_data]:
            mapped_data.append({
                "key_name": key,
                "data_path": "not found",
                "content": content,
            })
 
    return mapped_data[0]
 
#-----------------------------------------Summary---------------------------------------------------------------
 
def calculate_match_score_executive_summary(category_words, content_words):
    """Calculate a match score based on the overlap of words."""
    common_words = category_words.intersection(content_words)
    return len(common_words) / len(category_words) if category_words else 0  # Avoid division by zero
 
def map_categories_to_json_Executive_Summary(json_data):
    """
    Maps JSON keys to categories and handles special cases.
 
    Parameters:
        json_data (dict): Input JSON data containing text content.
 
    Returns:
        list: A list of dictionaries with key_name, data_path, and content.
    """
    print("json_Executive_Summary...", json_data)
    mapped_data = []  # List to store all mappings
 
    for key, content in json_data.items():
        # General case: Match content to categories
        best_match = None
        best_score = 0
 
        for category, data_path in DATA_PATH_MAPPING_EXECUTIVE_SUMMARY.items():
            normalized_category = normalize_text(category)
            normalized_content = normalize_text(content)
 
            # Tokenize category and content into words
            category_words = set(normalized_category.split())
            content_words = set(normalized_content.split())
 
            # Calculate match score
            score = calculate_match_score_executive_summary(category_words, content_words)
 
            # Update best match if this category has a higher score
            if score > best_score:
                best_match = {
                    "key_name": key,
                    "data_path": data_path,
                    "content": content,
                }
                best_score = score
 
        # Add the best match to the result
        if best_match and best_score > 0.5:  # Adjust threshold as needed
            mapped_data.append(best_match)
        else:
            # Fallback if no good match is found
            mapped_data.append({
                "key_name": key,
                "data_path": "not found",
                "content": content,
            })
 
    return mapped_data[0]


