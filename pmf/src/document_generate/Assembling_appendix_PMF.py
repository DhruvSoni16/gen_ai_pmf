import os
from PyPDF2 import PdfMerger
from fpdf import FPDF
from fuzzywuzzy import fuzz
from docx2pdf import convert
import comtypes.client  # For Excel to PDF conversion (Windows only)
import tempfile
import shutil
import pythoncom
import win32com.client
import os
import win32com.client
import pythoncom
import time
# Appendices configuration
appendices = {
    "Appendix A": {
        "description": "Site Google Map"
    },
    "Appendix B": {
        "description": "Organization Chart"
    },
    "Appendix C": {
        "description": "Plant layout"
    },
    "Appendix D": {
        "description": "List of Laboratory Equipment"
    },
    "Appendix E": {
        "description": "Asheville QMS Procedures relevant for this application"
    }
}

# Function to create a placeholder page for unavailable appendices
def create_not_available_page(appendix_name, description, output_file):
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
    pdf.close()

# Function to convert Word to PDF
def convert_word_to_pdf(doc_path, pdf_path):
    # Ensure absolute paths
    doc_path = os.path.abspath(doc_path)
    pdf_path = os.path.abspath(pdf_path)
 
    # Ensure clean COM threading
    pythoncom.CoInitialize()
 
    word_app = None
    doc = None
    try:
        word_app = win32com.client.DispatchEx("Word.Application")
        word_app.Visible = False
        word_app.DisplayAlerts = False
 
        doc = word_app.Documents.Open(doc_path)
        doc.ExportAsFixedFormat(pdf_path, ExportFormat=17)  # 17 = PDF
 
    except Exception as e:
        print(f"[ERROR] Failed to convert: {e}")
    finally:
        if doc:
            doc.Close(False)
        if word_app:
            word_app.Quit()
        # Release COM objects
        del doc
        del word_app
        pythoncom.CoUninitialize()

# Function to convert Excel to PDF
def convert_excel_to_pdf(input_file, output_file):
    try:
        input_file = os.path.abspath(input_file)
        output_file = os.path.abspath(output_file)

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        excel = comtypes.client.CreateObject("Excel.Application")
        excel.Visible = False
        workbook = excel.Workbooks.Open(input_file)
        workbook.Save()
        workbook.ExportAsFixedFormat(0, output_file)
        workbook.Close(SaveChanges=False)
        excel.Quit()
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting Excel to PDF: {e}")
        if 'excel' in locals():
            excel.Quit()

# Main function to assemble appendices
def assemble_appendices(base_folder, original_document, output_document):
    global appendices

    if not os.path.exists(base_folder):
        print(f"Base folder '{base_folder}' not found.")
        exit()

    merger = PdfMerger()

    if os.path.exists(original_document):
        merger.append(original_document)
    else:
        print(f"Original document '{original_document}' not found.")
        exit()

    temp_dir = tempfile.mkdtemp()
    available_files = os.listdir(base_folder)
    print('Available files:', available_files)

    for appendix, details in appendices.items():
        description = details["description"]
        print(f"Processing {appendix}: {description}")

        files_processed = False
        for file in available_files:
            file_path = os.path.join(base_folder, file)
            try:
                match_score = fuzz.partial_ratio(description.lower(), file.lower())
                if match_score >= 70:
                    print(f"Matched file: {file} (score: {match_score})")

                    if os.path.getsize(file_path) == 0:
                        print(f"Skipping empty file: {file_path}")
                        continue

                    # If a matching document is found, append it directly without creating a title page
                    if file.endswith(".pdf"):
                        merger.append(file_path)
                        files_processed = True
                    elif file.endswith(".doc") or file.endswith(".docx"):
                        converted_pdf = os.path.join(temp_dir, file.replace(".docx", ".pdf").replace(".doc", ".pdf"))
                        convert_word_to_pdf(file_path, converted_pdf)
                        merger.append(converted_pdf)
                        files_processed = True
                    elif file.endswith(".xls") or file.endswith(".xlsx"):
                        converted_pdf = os.path.join(temp_dir, file.replace(".xlsx", ".pdf").replace(".xls", ".pdf"))
                        convert_excel_to_pdf(file_path, converted_pdf)
                        merger.append(converted_pdf)
                        files_processed = True
                    else:
                        print(f"Unsupported file format: {file_path}")
            except Exception as e:
                print(f"Error processing file '{file_path}': {e}")

        # If no files were processed for this appendix, create a "Not Available" page
        if not files_processed:
            print(f"No suitable files found for {appendix}: {description}")
            not_available_page = os.path.join(temp_dir, f"{appendix}_NotAvailable.pdf")
            create_not_available_page(appendix, description, not_available_page)
            merger.append(not_available_page)

    merger.write(output_document)
    merger.close()
    shutil.rmtree(temp_dir)
    print(f"Final document with appendices created: {output_document}")
    return {"Status": "SUCCESS", "Result": "Final document with appendices created"}

# # Testing Scenarios
# if __name__ == "__main__":

#     base_folder = "C:\\Users\\naveen.jallepalli\\OneDrive - Thermo Fisher Scientific\\Documents\\TFS_POC\\POC_Prod\\testing_appendix_assembling\\Langsbold SOP"

#     # Define the original document
#     original_document = "C:\\Users\\naveen.jallepalli\\OneDrive - Thermo Fisher Scientific\\Documents\\TFS_POC\\POC_Prod\\testing_appendix_assembling\\DMF_Output_20250703_173858.pdf"

#     # Define the output file
#     output_document = "C:\\Users\\naveen.jallepalli\\OneDrive - Thermo Fisher Scientific\\Documents\\TFS_POC\\POC_Prod\\testing_appendix_assembling\\FinalDocumentWithAppendices_13.pdf"

#     # Test Case 1: Normal scenario with matching files
#     print("Test Case 1: Normal scenario with matching files")
#     res = assemble_appendices(base_folder, original_document, output_document)
#     print(res)

