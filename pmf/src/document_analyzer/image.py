import os
import json
import io
import logging
import time
import base64
import re
import shutil
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import PIL.Image
import openai
from docx import Document
from io import BytesIO
import streamlit as st

  # Assuming this is for another purpose not shown in the code

# Load the configuration file
with open('Config/configuration.json', 'r') as f:
    config = json.load(f)

# Set the OpenAI API key
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY', '')

logging.basicConfig(level=logging.INFO,filename='logs/app.log',filemode='a',format='%(asctime)s - %(levelname)s - %(message)s')

def clear_extracted_folder(folder_path):
    """
    Clears all files and subdirectories in the specified folder.
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    # Use shutil.rmtree with error handling for permissions
                    shutil.rmtree(file_path, onerror=handle_remove_readonly)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(folder_path)
# Function to extract image indices from a PDF based on keyword
def extract_image_indices(pdf_name, name, flag):
    """
    Extracts indices of pages containing a specific name or keyword from a PDF.
    """
    try:
        pdf_stream = pdf_name
        indices = []
        with pdfplumber.open(pdf_stream) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if flag == 0:
                    if text and name.lower() in text.lower():
                        indices.append(i)
                else:
                    if "Figure" in text and name.lower() in text.lower():
                        indices.append(i)

        return indices
    except Exception as e:
        logging.error(f"Error extracting image indices from PDF: {e}")
        return []

# Function to clean non-alphabetic text
def clean_text(text):
    """
    Cleans the input text by removing lines that do not contain any alphabetic characters.
    """
    try:
        lines = text.split("\n")
        cleaned_text = "\n".join([line for line in lines if any(char.isalpha() for char in line)]).strip()
        return cleaned_text
    except AttributeError as e:
        logging.error(f"Error processing text: {e}. Ensure the input is a string.")
        return ""
    except Exception as e:
        logging.error(f"Unexpected error in clean_text function: {e}")
        return ""

# Function to extract the text above the figure and the figure captions
def extract_text_parts(pdf_name, indices,flag):
    image_data = []
    if flag==0:
        figure_caption_pattern = re.compile(r"(Figure\s\d+\..+?)(?=\n|$)")
    else:

        figure_caption_pattern = re.compile(r"(Figure\s\d+[:.]?\s?.+?)(?=\n|$)", re.MULTILINE)

    pdf_stream = pdf_name
    with pdfplumber.open(pdf_stream) as pdf:
        for index in indices:
            page = pdf.pages[index]
            text = page.extract_text()
            text = clean_text(text)

            captions = figure_caption_pattern.findall(text)
            for caption in captions:
                caption_index = text.find(caption)
                text_before_caption = text[:caption_index].strip()
                lines_before_caption = text_before_caption.split("\n")
                above_text = lines_before_caption[-1].strip() if lines_before_caption else ""

                image_data.append({
                    "above_text": above_text,
                    "fig_caption": caption.strip(),
                    "page_index": index
                })

    return image_data


# Function to extract the graphical region (Code 1)
def extract_graphical_region_from_pdf(pdf_path, target_heading, figure_caption, output_folder, page_number,flag):
    
    # pdf_stream = BytesIO(pdf_path)
    pdf_document = fitz.open(pdf_path)
   
    heading_rect, caption_rect = None, None
    page = pdf_document.load_page(page_number)

     # Search for heading and caption text
    heading_instances = page.search_for(target_heading)
    caption_instances = page.search_for(figure_caption)

    if heading_instances:
        heading_rect = heading_instances[0]  # Use the first instance of heading
    if caption_instances:
        caption_rect = caption_instances[0]  # Use the first instance of caption

    # Ensure both heading and caption are found
    if heading_rect and caption_rect:
        # Calculate the exact bounds between heading and caption
        upper_bound_y = heading_rect.y1
        lower_bound_y = caption_rect.y0
        graphical_rects = []

        # Get any drawings and images within the bounds
        for drawing in page.get_drawings():
            drawing_rect = fitz.Rect(drawing["rect"])
            if upper_bound_y < drawing_rect.y0 < lower_bound_y:
                graphical_rects.append(drawing_rect)

        for img in page.get_images(full=True):
            img_rect = page.get_image_bbox(img)
            if upper_bound_y < img_rect.y0 < lower_bound_y:
                graphical_rects.append(img_rect)

        # Proceed if we have graphical regions within bounds
        if graphical_rects:
            # Calculate the bounding box with no padding
            min_x = min(rect.x0 for rect in graphical_rects)
            max_x = max(rect.x1 for rect in graphical_rects)
            min_y = upper_bound_y
            max_y = lower_bound_y

            # Define the region with the exact bounds
            region_rect = fitz.Rect(min_x, min_y, max_x, max_y)
            pix = page.get_pixmap(clip=region_rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Save the image to the output folder
            os.makedirs(output_folder, exist_ok=True)
            output_image_path = os.path.join(output_folder, f"img_{page_number + 1}.png")
            img.save(output_image_path)
            pdf_document.close()
            return True  # Image extraction was successful

    pdf_document.close()
    return False  # No images extracted

# Fallback image extraction logic (Code 2)
def extract_images_from_pdf(pdf_path, output_folder, indices):
    pdf_stream = BytesIO(pdf_path)
    pdf_document = fitz.open(stream=pdf_path,filetype="pdf")

    if os.path.exists(output_folder):
        clear_extracted_folder(output_folder)
        # shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for page_number in indices:
        page = pdf_document.load_page(page_number)
        image_list = page.get_images(full=True)

        if image_list:
            zoom = 2  # Adjust resolution
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            temp_image_path = f"page_{page_number + 1}.png"
            pix.save(temp_image_path)
            rendered_img = Image.open(temp_image_path)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_rect = page.get_image_rects(xref)[0]
                left, upper, right, lower = image_rect

                left = max(left , 0)
                upper = max(upper , 0)
                right = min(right , page.rect.width)
                lower = min(lower , page.rect.height)

                cropped_img = rendered_img.crop((left * zoom, upper * zoom, right * zoom, lower * zoom))
                output_image_path = os.path.join(output_folder, f"img_{img_index + 1}_{page_number + 1}.png")
                cropped_img.save(output_image_path)

    pdf_document.close()

# Function to extract the text above the figure and the figure captions[COLUMN]
def extract_text_parts_col(pdf_name, indices, num_columns=2):
    """Extracts text above figures and figure captions from specified pages, handling column layouts."""
    image_data = []
    figure_caption_pattern = re.compile(r"(Figure\s\d+[:.]?\s?.+?)(?=\n|$)", re.MULTILINE)  # Regex for figure captions
    
    pdf_stream = pdf_name
    with pdfplumber.open(pdf_stream) as pdf:
        for index in indices:
            page = pdf.pages[index]
            
            # Split the page into columns
            page_width = page.width
            column_width = page_width / num_columns
            for col in range(num_columns):
                left_boundary = col * column_width
                right_boundary = (col + 1) * column_width

                # Crop the page to the column region and extract text from that column
                column_crop = (left_boundary, 0, right_boundary, page.height)
                column_text = page.within_bbox(column_crop).extract_text(layout=True)
                column_text = clean_text(column_text)

                # Search for figure captions in the extracted column text
                captions = figure_caption_pattern.findall(column_text)

                for caption in captions:
                    caption_index = column_text.find(caption)
                    text_before_caption = column_text[:caption_index].strip()
                    lines_before_caption = text_before_caption.split("\n")

                    above_text = lines_before_caption[-1].strip() if lines_before_caption else ""

                    if re.match(r"(Figure\s\d+[:.]?\s?.+?)(?=\n|$)", caption):
                        figure_caption = caption.strip()
                    else:
                        continue  # Skip partial or invalid captions

                    if figure_caption and above_text:
                        image_data.append({
                            "above_text": above_text,
                            "fig_caption": figure_caption,
                            "page_index": index,
                            "column": col
                        })
    return image_data

# secondary logic or column logical
def extract_graphical_region_second_method(pdf_path, target_heading, figure_caption, output_folder, page_number, column, num_columns=2):
    """Extracts graphical content between a specified heading and figure caption within a specific column."""
    print("Fallback or column level logic work")

    
    # pdf_stream = BytesIO(pdf_path)
    pdf_document = fitz.open(pdf_path)

    
    page = pdf_document.load_page(page_number)

    # Calculate column boundaries
    page_width = page.rect.width
    column_width = page_width / num_columns
    left_boundary = column * column_width
    right_boundary = (column + 1) * column_width

    # Search for the heading and caption within the column
    heading_instances = page.search_for(target_heading)
    caption_instances = page.search_for(figure_caption)

    # Filter instances within the column's bounding box
    heading_instances = [inst for inst in heading_instances if left_boundary <= inst.x0 <= right_boundary]
    caption_instances = [inst for inst in caption_instances if left_boundary <= inst.x0 <= right_boundary]

    if heading_instances and caption_instances:
        heading_rect = heading_instances[0]
        caption_rect = caption_instances[0]

        print(f"Extracting region between heading at {heading_rect} and caption at {caption_rect} in column {column}")

        graphical_rects = []

        # Get drawings and images from the page
        drawings = page.get_drawings()
        images = page.get_images(full=True)

        for drawing in drawings:
            drawing_rect = fitz.Rect(drawing["rect"])
            if heading_rect.y1 < drawing_rect.y0 < caption_rect.y0 and left_boundary <= drawing_rect.x0 <= right_boundary:
                graphical_rects.append(drawing_rect)

        for img in images:
            img_rect = page.get_image_bbox(img)
            if heading_rect.y1 < img_rect.y0 < caption_rect.y0 and left_boundary <= img_rect.x0 <= right_boundary:
                graphical_rects.append(img_rect)

        if graphical_rects:
            # Merge graphical rectangles to get the total width and bounding box
            min_x = max(left_boundary, min(rect.x0 for rect in graphical_rects))
            max_x = min(right_boundary, max(rect.x1 for rect in graphical_rects))
            min_y = min(rect.y0 for rect in graphical_rects)
            max_y = max(rect.y1 for rect in graphical_rects)

            # Define the region for each column separately
            region_rect = fitz.Rect(min_x, min_y, max_x, max_y)

            # Generate pixmap from the region
            pix = page.get_pixmap(clip=region_rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Ensure the output folder exists
            os.makedirs(output_folder, exist_ok=True)
            output_filename = os.path.join(output_folder, f"page_{page_number + 1}.png")
            
            # Save the extracted region as an image
            img.save(output_filename)
            print(f"Saved extracted region: {output_filename}")
            
            pdf_document.close()
            return output_filename  # Return the filename if extraction is successful

        else:
            print("No graphical content found between the heading and caption.")
            pdf_document.close()
            return None  # Return None if no graphical content is found

    else:
        print(f"Could not find both heading and caption in column {column} on page {page_number + 1}.")
        pdf_document.close()
        return None  # Return None if either heading or caption is not found

# Integrated function to apply both logic
def extract_images_with_fallback(pdf_path, output_folder,name,flag):
    
    if os.path.exists(output_folder):
        clear_extracted_folder(output_folder)
        # shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    indices = extract_image_indices(pdf_path,name,flag)
    st.write("Indices of pages containing the keyword:", indices)
    print(indices)
    image_data = extract_text_parts(pdf_path, indices,flag)

    images_extracted = False
    for data in image_data:
        target_heading = data["above_text"]
        figure_caption = data["fig_caption"]
        page_index = data["page_index"]

        # Try extracting images with Code 1 logic
        images_extracted = extract_graphical_region_from_pdf(pdf_path, target_heading, figure_caption, output_folder, page_index,flag)

        if not images_extracted:

            image_data = extract_text_parts_col(pdf_path, indices)

            for data in image_data:
                target_heading = data["above_text"]
                figure_caption = data["fig_caption"]
                page_number = data["page_index"]
                column=data["column"]

            images_extracted = extract_graphical_region_second_method(pdf_path, target_heading, figure_caption, output_folder,page_number, column, num_columns=2)


        # If Code 1 fails, use Code 2 logic as a fallback
        if not images_extracted:
            print(f"Fallback: Extracting images from page {page_index + 1} using Code 2 logic.")
            extract_images_from_pdf(pdf_path, output_folder, [page_index])
        
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except (FileNotFoundError, IOError) as e:
        logging.error(f"Error opening or reading the file: {e}")
        return None

def image_selection_1(folder_name,name):
    image_paths=[]
    image_name=[]
    for f in os.listdir(folder_name):

        image_name.append(f)
        image_paths.append(f"{folder_name}\\{f}")
    if len(image_name)>0:
    
        messages = [
            {"role": "system", "content": f"You are a helpful assistant that responds in Markdown. Help me in finding {name} images and give only image index"}
        ]

        for idx, image_path in enumerate(image_paths, start=1):
            base64_image = encode_image(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"This is image {idx}, is this a {name} image? give only index"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            })
        
        

        start_time = time.time()
        response = openai.ChatCompletion.create(
            model=config["model_name"],  # Replace with your model
            messages=messages,
            temperature=0.5,
        )   
        response_time = time.time() - start_time

    # Print the response
        response_content= response.choices[0].message.content

        if response and response['choices']:
            input_tokens = response['usage']['prompt_tokens']
            output_tokens = response['usage']['completion_tokens']
            total_tokens = response['usage']['total_tokens']
            logging.info(f"Image Extraction Section.................")
            logging.info(f"Input Tokens: {input_tokens}")
            logging.info(f"Output Tokens: {output_tokens}")
            logging.info(f"Total Tokens: {total_tokens}")
            logging.info(f"Response generation time: {response_time:.2f} seconds")
        match = re.search(r'\d+', response_content)
        if match:
            index_selection = int(match.group(0))  # Convert matched string to an integer
            selected_image = image_name[index_selection - 1]
            print("------------------------------------------------------------")
            
            return selected_image,{
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'response_time': response_time
        }
        else:
            logging.error("No index found in response: " + response_content)
            return None,None
    else:
        return None,None

# Function to generate a final output based on GPT-4 response
def final_image_output_GPT(text, reference_content):
    prompt = f"""
           You must refer strictly to the reference template provided below and perform the following tasks:
    
        - **Do not include** any acknowledgments of the model's capabilities or limitations.
        - Focus exclusively on the "1.Image extraction"  section within the reference template,while excluding any 'Text Extraction' or 'Table Extraction.
        - Do not repeat any sections from the reference template or add any additional information.
        - Do not print any conversational responses generated by the model.
        - response shoulde be in order as in given "Content to Process"
        Reference Template:\n{reference_content}\n
        Content to Process:\n{text}   

"""
        
    start_time = time.time()

    response = openai.ChatCompletion.create(
        model=config['model_name'],  # Use GPT-4 or GPT-4-turbo based on your configuration
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=config['generation_config']['temperature'],
        max_tokens=config['generation_config']['max_tokens'],
        top_p=config['generation_config']['top_p'],
    )

    response_time = time.time() - start_time

    # Log the token usage and response time
    if response and response['choices']:
        input_tokens = response['usage']['prompt_tokens']
        output_tokens = response['usage']['completion_tokens']
        total_tokens = response['usage']['total_tokens']
        
        logging.info(f"Image Description Section.................")
        logging.info(f"Input Tokens: {input_tokens}")
        logging.info(f"Output Tokens: {output_tokens}")
        logging.info(f"Total Tokens: {total_tokens}")
        logging.info(f"Response generation time: {response_time:.2f} seconds")

        return response['choices'][0]['message']['content'], {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'response_time': response_time
        }
    else:
        return None,None
    

def final_image_output_GPT_cer(text, reference_content):
    prompt = f"""This is the context: {text} and here is the output template: {reference_content}.
        Your task is to strictly follow the reference template and FOCUS SOLELY ON THE 'Image Extraction' section of the reference template and extract the Refrigeration System image or image of the Device provided and print it in the 'Image Extraction' section.  
        **Do not include** any acknowledgments of the model's capabilities or limitations. 
            
           """ 
    start_time = time.time()

    response = openai.ChatCompletion.create(
        model=config['model_name'],  # Use GPT-4 or GPT-4-turbo based on your configuration
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=config['generation_config']['temperature'],
        max_tokens=config['generation_config']['max_tokens'],
        top_p=config['generation_config']['top_p'],
    )

    response_time = time.time() - start_time

    # Log the token usage and response time
    if response and response['choices']:
        input_tokens = response['usage']['prompt_tokens']
        output_tokens = response['usage']['completion_tokens']
        total_tokens = response['usage']['total_tokens']
        
        logging.info(f"final_image_output_GPT_cer Section.....................")
        logging.info(f"Input Tokens: {input_tokens}")
        logging.info(f"Output Tokens: {output_tokens}")
        logging.info(f"Total Tokens: {total_tokens}")
        logging.info(f"Response generation time: {response_time:.2f} seconds")

        return response['choices'][0]['message']['content'], {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'response_time': response_time
        }
    else:
        return None,None