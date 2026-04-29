import fitz
import re
import openai
import time
import logging
import json
from docx import Document 
from pdf2image import convert_from_bytes
from pdf2image import convert_from_path
import stat
from docx import Document
# from IPython.display import Image, display
import base64
import shutil
import cv2
import numpy as np
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
import pytesseract
import zipfile
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'


# Load the config file
with open('Config/configuration.json', 'r') as f:
    config = json.load(f)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY', '')
api = os.getenv('OPENAI_API_KEY', '')

logging.basicConfig(
    filename='logs/app.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def on_rm_error(func, path, exc_info):
    # Change the permissions and retry
    os.chmod(path, stat.S_IWRITE)
    func(path)
# convert pdf page to images
def pdf_to_images(pdf_bytes):
    """
    Converts a PDF file (in bytes) to a list of images, one for each page.
    """
    try:
        doc = fitz.open(pdf_bytes)
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            images.append(image)
        return images
    except Exception as e:
        print("error in pdf_to_image")
        print(e)
        return []
    

# yellow color detection in page
def contains_yellow(image):
    """
    Checks if the given image contains yellow color.
    """
    try:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        yellow_mask = cv2.inRange(hsv_image, (20, 100, 100), (30, 255, 255))
        return np.any(yellow_mask)
    except cv2.error as e:
       
        return False
    except Exception as e:
       
        return False


def extract_images_and_figures_page_number(folder_path,output_folder):
    # global image_list
    folder_path=folder_path[0]
    st.write("EP-1")
    # st.write(folder_path)
    # extract all files from input folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    pdf_path = fr"{folder_path}/{files[0]}"
    # st.write("pdf_path"+pdf_path)

    images =  pdf_to_images(pdf_path)
    # st.write(images)
    
    
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    pages_chosen = []
    
    for page_num, image in enumerate(images):
        page = reader.pages[page_num]
        text = page.extract_text().lower()
        
        keywords=["hazard","warning","caution","Precaution"]
        if contains_yellow(image) or any(word in text.lower() for word in keywords):
            writer.add_page(page)
            pages_chosen.append(page_num+1)

    # if os.path.exists(output_folder):
        # shutil.rmtree(output_folder,onerror=on_rm_error)
    os.makedirs(output_folder, exist_ok=True)
    
    # image_list = convert_from_path(pdf_path)    
    for page in pages_chosen:
        # PDF pages are 1-indexed
        images_from_page = convert_from_path(pdf_path, first_page=page, last_page=page,poppler_path=r"C:\poppler-24.08.0\Library\bin")
    
        image_path = f"{output_folder}/page_{page}.png"
        images_from_page[0].save(image_path, 'JPEG')


# encode the image to base64 
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        # logging.error(f"Error opening image file {image_path}: {e}")
        return None
    
def image_based_warning(folder_name,llm,client):
    image_paths=[]
    image_name=[]
    for f in os.listdir(folder_name):
        image_name.append(f)
        image_paths.append(f"{folder_name}\\{f}")
        # ]
    image_text=[]
    input_tokens=0
    output_tokens=0
    total_tokens=0
    for idx, image_path in enumerate(image_paths, start=0):
            messages = [
                {
            "role": "system",
            "content": """Analyze the following text and extract all **WARNINGS,NOTE,IMPORTANT NOTE and CAUTIONS with there cateory:[General Recommendations,Installation,Ice Scraper Instruction,Chart Recorders - If installed (Optional),Backup System -If installed (Optional),Maintenance,Safety Considerations]**. If some warnings or cautions do not **explicitly use these keywords but imply risks, hazards, or safety precautions, extract those all statements as well**. Focus only on the text that includes a yellow triangle symbol on the left-hand side or any hazard-related symbols. **Note:- 1)Exclude any predefined template sentences that describe what symbols indicate, such as warnings about electrical shock, fire, sharp points, hot surfaces, gloves, or pinch points. 2) Do not add any extra information or comments—respond only with the extracted text.\n3) If no relevant content is found, return an empty response without any comment or apology**. Ensure the extraction is precise and captures all risk-related information associated with those symbols.
                        --Do not add assume categories—categorize Maintain clarity and precision in extraction, ensuring accurate placement under predefined categories. 
                        """
            },]
            base64_image = encode_image(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": """Analyze the following text and extract all **WARNINGS,NOTE,IMPORTANT NOTE and CAUTIONS** with there cateory:[General Recommendations,Installation,Ice Scraper Instruction,Chart Recorders - If installed (Optional),Backup System -If installed (Optional),Maintenance,Safety Considerations] . If some warnings or cautions do not explicitly use these keywords but imply risks, hazards, or safety precautions, extract those statements as well. 
                              extract and focus only on the text that includes a yellow triangle symbol on the left-hand side or any hazard-related symbols. **Note:- Exclude any predefined template sentences that describe what symbols indicate, such as warnings about electrical shock, fire, sharp points, hot surfaces, gloves, or pinch points** and **Do not add any extra information or comments—respond only with the extracted text **, Ensure the extraction is precise and captures all risk-related information associated with those symbols.
                            -Do not add assume categories—categorize Maintain clarity and precision in extraction, ensuring accurate placement under predefined categories. 
                     """},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            })

            
            response = client.chat.completions.create(
                    model="gpt-4o",  # Replace with your model
                    messages=messages,
                    temperature=0.0,
                )  
            
            response_dict = response.model_dump()
            image_text.append(response_dict["choices"][0]["message"]["content"])

   
                    
    return image_text


def process_warning_text_with_GPT(text, reference_content,llm,client):
    """
    Sends the extracted text to the OpenAI API for processing based on the reference content provided.
    """
    prompt = f"""This is the "context": {text}.
        **write heading on the top Warning and Precaution **  
        Your task is to extract **all information** from the provided list, focusing solely on meaningful content, including warnings, cautions, important note and instructions.  

        **Instructions for Extraction:**  
        1. **Extract all warnings and cautions** (whether explicitly labeled or implied) and any other relevant safety instructions.  
        2. **Exclude**:
        - **Symbol indications** (e.g., ⚠️, hazard triangles).
        - **Model or system capability-related statements**, such as: "I'm sorry, I can't assist with that."
        3. **Do not extract any statments from the legend which contains the word "This symbol indicates".**
        4. must**Do not include any acknowledgments of the model's capabilities or limitations**
        5. **Maintain the original structure** of the text and return **each item in its complete form**.  
        6. **Do not alter, summarize, or omit** any part of the extracted content. Keep all relevant warnings, cautions, and instructions intact.
        
        
        Focus on clear and comprehensive extraction following the above rules.

        
        Reference Template:\n{reference_content}\n
       
        """


    response = client.chat.completions.create(
        model="gpt-4o",  # Use GPT-4 or GPT-4-turbo based on your configuration
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
       
    )
    response_dict = response.model_dump()
    return response_dict['choices'][0]['message']['content']
 







def extract_images_from_docx(docx_path, output_dir="extracted_images"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    docx_path=docx_path[0]
    # Open the .docx file as a zip archive
    with zipfile.ZipFile(docx_path, 'r') as docx_zip:
        # Look for image files inside 'word/media/'
        for file in docx_zip.namelist():
            if file.startswith('word/media/'):
                filename = os.path.basename(file)
                file_path = os.path.join(output_dir, filename)

                # Extract and save each image
                with open(file_path, 'wb') as f:
                    f.write(docx_zip.read(file))

    return output_dir

def image_based_manufecturing(folder_name,llm, client):
    image_paths=[]
    image_name=[]
    image_paths1=[]
    for f in os.listdir(folder_name):
        image_name.append(f)
        image_paths.append(f"{folder_name}\\{f}")
    image_paths1.append(image_paths[0])
        # ]
    image_text=[]
    input_tokens=0
    output_tokens=0
    total_tokens=0
    for idx, image_path in enumerate(image_paths1, start=0):
            messages = [
                {
            "role": "system",
            "content": f"""
                    Introduction: Begin by explaining the importance of the manufacturing process. Highlight how it helps in producing high-quality products in an efficient manner, ensuring consistency and reliability.

                    Describe each step of the manufacturing process, one by one. Make sure to:
                    -Manufacturing Processes flow step given in the image you have to strictly follow this sequence and each and every step should be in extracted from the image.do not ommiting any step ot part wich are use in processing of manufacturing process.
                    -Clearly explain what happens in each step.
                    Use simple, easy-to-understand language.
                    Flow Sequence: Write very detailed descriptions for each step, ensuring the sequence of operations is maintained.

                    Clearly explain what happens in each step.
                    Use simple, easy-to-understand language. 
                    Flow Sequence: Write very detailed descriptions for each step, ensuring the sequence of operations is maintained.

                    **Provide Details: Describe each step minimum in 3/4 lines. **.
                    **do not generate main step and then sub step**
                    - do not add ### or # before step just bod using ** **
                    By following these instructions, you can create a clear, thorough summary that captures the essence of the manufacturing process, step by step.
                    
                        """
            },]
            base64_image = encode_image(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
                        Introduction:  Begin by add heading "**4.2 Manufacturing Processes**" and explaining the importance of the manufacturing process. Highlight how it helps in producing high-quality products in an efficient manner, ensuring consistency and reliability.

                    Describe each step of the manufacturing process, one by one. Make sure to:
                   -Manufacturing Processes flow step given in the image you have to strictly follow this sequence and each and every step should be in extracted from the image.do not ommiting any step ot part wich are use in processing of manufacturing process.

                    Clearly explain what happens in each step.
                    Use simple, easy-to-understand language.
                    Flow Sequence: **Write very detailed descriptions for each step**, ensuring the sequence of operations is maintained.

                    **Provide Details: Describe each step minimum 50 to 60 words.**.
                     - do not add ### or # before step just bod using ** **
                    **do not generate main step and then sub step**
                     
                    By following these instructions, you can create a clear, thorough summary that captures the essence of the manufacturing process, step by step.
                    
                     """},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            })

            
            response = client.chat.completions.create(
                    model="gpt-4o",  # Replace with your model
                    messages=messages,
                    temperature=0.4,
                )  
            
            response_dict = response.model_dump()
            image_text.append(response_dict["choices"][0]["message"]["content"])

   
                    
    return image_text


def process_text_with_GPT(text, reference_content, llm,client):
    
    # Define your prompt template with placeholders
    prompt_template = """
        **don't include in answer : provide text, given text, Based on the provided reference template and the PDF text,**
        Your task is to extract all the information from the provided text and based on the reference template provided.
        Include all relevant details, ensuring that the output is structured and follows the reference template closely.like heading, subheading, and any other relevant information.
        **do not add any extra information or comments—respond only with the extracted text.**
        task:

        Inputs:-

        Reference Template:
        {reference_content}

        Input Text:
        {text}
            """

    # Create a prompt template instance
    prompt = PromptTemplate(template=prompt_template, input_variables=["text", "reference_content"])

    # Format the prompt with input variables
    formatted_prompt = prompt.format(text=text, reference_content=reference_content)

    # For AzureChatOpenAI (chat model), you need to wrap the prompt into messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=formatted_prompt)
    ]

    # Invoke the model
    response = llm.invoke(messages)

    # Return the content text from the response
    return response.content