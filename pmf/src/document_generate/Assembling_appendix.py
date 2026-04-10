import os
from PyPDF2 import PdfMerger
from fpdf import FPDF
from fuzzywuzzy import process
import comtypes.client  # For Word to PDF conversion (Windows only)
import tempfile
import shutil
from docx2pdf import convert
import pythoncom


appendices={
    "Appendix A": {
        "description": "Instruction for Use/User Manual",
        "possible_folders": ["User Manual"]  # No variations
    },
    "Appendix B": {
        "description": "Degree of Novelty Card",
        "possible_folders": []  # No variations
    },
    "Appendix C": {
        "description": "Safety Test Report",
        "possible_folders": ["Safety report"]  # No variations
    },
    "Appendix D": {
        "description": "EMI/EMC Test Report",
        "possible_folders": ['EMC']  # No variations
    },
    "Appendix E": {
        "description": "Technical Data Sheet, Product Brochure and Labels",
        "possible_folders": ["TDS , Technical Data", "Brochure", "Labels"]  # Example variations
    },
    "Appendix F": {
        "description": "Design Process Flow Chart",
        "possible_folders": []  # Example variations
    },
    "Appendix G": {
        "description": "Verification & Validation Test Reports",
        "possible_folders": ["V&V , Verification and Validation Reports"]  # Example variations
    },
    "Appendix H": {
        "description": "Manufacturing Process Flow Chart",
        "possible_folders": []  # Example variations
    },
    "Appendix I": {
        "description": "Essential Principles Checklist",
        "possible_folders": []  # Example variations
    },
    "Appendix J": {
        "description": "Risk Management file",
        "possible_folders": ["Risk Management files"]  # Example variations
    },
    "Appendix K": {
        "description": "Software Verification & Validation Reports",
        "possible_folders": ["Software V&V , Software Verification & Validation Reports"]  # Example variations
    },
    "Appendix L": {
        "description": "Clinical Evaluation Plan",
        "possible_folders": []  # Example variations
    },
    "Appendix M": {
        "description": "Clinical Evaluation Report",
        "possible_folders": []  # Example variations
    },
    "Appendix N": {
        "description": "Post-Market Surveillance Plan",
        "possible_folders": []  # Example variations
    },
    "Appendix O": {
        "description": "Final Inspection Certificate",
        "possible_folders": ["ISO 13485"]  # Example variations
    }
}


# def appendixes():
#     return appendices


# Function to create a title page for each appendix
def create_title_page(appendix_name, description, output_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=f"{appendix_name}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(200, 10, txt=f"{description}", align="C")
    pdf.output(output_file)
    pdf.close()  # Ensure the file is properly closed

# Function to create a placeholder page for unavailable appendices
def create_not_available_page(appendix_name,description, output_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=f"{appendix_name}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"{description}", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt="<Not available>", ln=True, align="C")
    pdf.output(output_file)
    pdf.close()  # Ensure the file is properly closed


def convert_word_to_pdf(input_file, output_file):
    """Convert a Word document to PDF using docx2pdf."""
    try:
        # Initialize COM in the current thread
        pythoncom.CoInitialize()
        convert(input_file, output_file)
        print(f"Converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting Word to PDF: {e}")
    finally:
        # Uninitialize COM to clean up
        pythoncom.CoUninitialize()
        

def convert_excel_to_pdf(input_file, output_file):
    """
    Convert an Excel file to PDF using comtypes.
    """
    try:
        # Convert to absolute paths
        input_file = os.path.abspath(input_file)
        output_file = os.path.abspath(output_file)

        # Debugging: Print the paths
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")

        # Check if the input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Create an instance of Excel
        excel = comtypes.client.CreateObject("Excel.Application")
        excel.Visible = False  # Run Excel in the background

        # Open the Excel file
        workbook = excel.Workbooks.Open(input_file)

        # Save the workbook before exporting
        workbook.Save()

        # Export as PDF
        workbook.ExportAsFixedFormat(0, output_file)  # 0 is the PDF format

        # Close the workbook and quit Excel
        workbook.Close(SaveChanges=False)
        excel.Quit()

        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting Excel to PDF: {e}")
        if 'excel' in locals():
            excel.Quit()


def assemble_appendices(base_folder, original_document, output_document):#, appendices):

    global appendices

    # Ensure base folder exists
    if not os.path.exists(base_folder):
        print(f"Base folder '{base_folder}' not found.")
        exit()

    # Create a PdfMerger object
    merger = PdfMerger()

    # Append the original document
    if os.path.exists(original_document):
        merger.append(original_document)
    else:
        print(f"Original document '{original_document}' not found.")
        exit()

    # Create a temporary directory for storing temporary files
    temp_dir = tempfile.mkdtemp()

    # Get all folder names in the base folder
    available_folders_1 = os.listdir(base_folder)
    available_folders = []
    for file in available_folders_1:
        available_folders.append(file.lower())

    print('available_folders...',available_folders)

    for appendix, details in appendices.items():
        description = details["description"]
        possible_folders = details["possible_folders"]

        print(f"Processing {appendix}: {description}")
        print(f"Possible folders for {appendix}: {possible_folders}")

        matched_folders = []

        if description == "Manufacturing Process Flow Chart":
            print('entered manufacturing process flow chart...')
            title_page_created = False  # Track if the title page has been created

            for file in available_folders:
                print('file...', file)
                if file.lower().startswith("sop") and file.lower().endswith(".docx"):
                    converted_pdf = os.path.join(temp_dir, file.replace(".docx", ".pdf"))
                    file_path = os.path.join(base_folder, file)
                    
                    try:
                        # Convert .docx to .pdf
                        convert_word_to_pdf(file_path, converted_pdf)
                        
                        # Create a title page only once
                        if not title_page_created:
                            title_page = os.path.join(temp_dir, f"{appendix}_TitlePage.pdf")
                            create_title_page(appendix, description, title_page)
                            merger.append(title_page)
                            title_page_created = True

                        # Append the converted PDF file
                        merger.append(converted_pdf)
                        print(f"Added converted SOP file: {converted_pdf}")
                    except Exception as e:
                        print(f"Error converting {file_path} to PDF: {e}")

                elif file.lower().startswith("sop") and file.lower().endswith(".pdf"):
                    sop_file_path = os.path.join(base_folder, file)
                    print('before replace sop_file_path...', sop_file_path)

                    # Create a title page only once
                    if not title_page_created:
                        title_page = os.path.join(temp_dir, f"{appendix}_TitlePage.pdf")
                        create_title_page(appendix, description, title_page)
                        merger.append(title_page)
                        title_page_created = True

                    # Append the SOP PDF file
                    merger.append(sop_file_path)
                    print(f"Added SOP file: {sop_file_path}")
            continue

        if possible_folders:
            for folder in possible_folders:
                # Apply filtering logic for "Verification & Validation Test Reports"
                if description == "Verification & Validation Test Reports":
                    filtered_folders = [
                        folder for folder in available_folders if "software" not in folder.lower()
                    ]
                else:
                    filtered_folders = available_folders  # Use all available folders for other descriptions

                print('folder .lower()...', folder.lower())
                print('filtered_folders...', [f.lower() for f in filtered_folders])

                # Split the folder name by commas and normalize each part
                parts = [part.strip().lower() for part in folder.split(",")]

                for part in parts:
                    # Check for direct matches
                    if part in [f.lower() for f in filtered_folders]:
                        print('Matched folder (direct):', part)
                        matched_folders.append(part)
                        break  # Stop checking other parts if a match is found
                    else:
                        # Fallback to fuzzy matching
                        print(f"Part '{part}' not found. Attempting fuzzy matching.")
                        best_match, score = process.extractOne(part, filtered_folders)
                        if score >= 90:  # Use a high threshold to ensure close matches
                            print(f"Fuzzy matched folder: {best_match} (score: {score})")
                            matched_folders.append(best_match)
                            break  # Stop checking other parts if a match is found
                        else:
                            print(f"No suitable match found for part '{part}'.")

        if matched_folders:
            # Deduplicate matched folders
            matched_folders = list(set(matched_folders))
            print(f"Matched folders for {appendix}: {matched_folders}")
            
            # Create a title page in the temporary directory
            title_page = os.path.join(temp_dir, f"{appendix}_TitlePage.pdf")
            create_title_page(appendix, description, title_page)
            merger.append(title_page)

            # Process files from all matched folders
            for matched_folder in matched_folders:
                folder_path = os.path.join(base_folder, matched_folder)
                
                # Recursively process all files in the folder and its subfolders
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        print('file...', file)  
                        file_path = os.path.join(root, file)
                        try:
                            # Check if the file is empty
                            if os.path.getsize(file_path) == 0:
                                print(f"Skipping empty file: {file_path}")
                                continue

                            if file.endswith(".msg"):
                                continue  # Skip .msg files
                            if file.endswith(".pdf"):
                                merger.append(file_path)
                            elif file.endswith(".doc") or file.endswith(".docx"):
                                converted_pdf = os.path.join(temp_dir, file.replace(".docx", ".pdf").replace(".doc", ".pdf"))
                                convert_word_to_pdf(file_path, converted_pdf)
                                merger.append(converted_pdf)
                            elif file.endswith(".xls") or file.endswith(".xlsx"):
                                # Skip .xls and .xlsx files
                                continue
                            else:
                                print(f"Unsupported file format: {file_path}")
                        except Exception as e:
                            print(f"Error processing file '{file_path}': {e}")
        else:
            print(f"No suitable match found for {appendix}: {description}")
            not_available_page = os.path.join(temp_dir, f"{appendix}_NotAvailable.pdf")
            create_not_available_page(appendix, description, not_available_page)
            merger.append(not_available_page)  
                    
    # Write the merged document
    merger.write(output_document)
    merger.close()

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)

    print(f"Final document with appendices created: {output_document}")

    return {"Status":"SUCCESS","Result":"Final document with appendices created"}

############################################

# from Assembling_appendix import assemble_appendices

# # Define the base folder path
# base_folder = "testing_appendix_assembling\STP ULT"

# # Define the original document
# original_document = "testing_appendix_assembling\output_1.pdf"  # Replace with the path to your original document

# # Output file
# output_document = "testing_appendix_assembling\FinalDocumentWithAppendices_3.pdf"

# res=assemble_appendices(base_folder, original_document, output_document)
# print(res)
