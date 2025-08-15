from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# Convert PDF pages to images
pdf_path = "Iceberg a la Vista (cap. 5-10).pdf"
images = convert_from_path(pdf_path)

# Perform OCR on each page and collect the text
extracted_text = []
for i, image in enumerate(images):
    text = pytesseract.image_to_string(image, lang='spa')  # specifying Spanish for OCR
    extracted_text.append(text)

# Joining all text for analysis
full_text = "\n".join(extracted_text)
full_text[:1000]  # Displaying only the first 1000 characters for a quick review
