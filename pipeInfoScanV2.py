import pytesseract
from PIL import Image
import pandas as pd
import cv2
import os

# Specify Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to the image
image_path = r'C:\Users\julia\PycharmProjects\IvanPipeProject\Scans\Scan2024-12-19_095936.jpg'

# Preprocess the image for better OCR results
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Binarization
    _, thresh_img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    # Resize for better OCR accuracy
    resized_img = cv2.resize(thresh_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return resized_img

# Perform OCR on specific regions
def extract_text_from_image(image, regions):
    data = {}
    for field, roi in regions.items():
        x, y, w, h = roi
        cropped = image[y:y+h, x:x+w]
        # Debug: Show cropped region (optional)
        if field == 'Service Address':
            cv2.imshow(f'{field}', cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # OCR
        text = pytesseract.image_to_string(cropped, config='--psm 6')
        data[field] = text.strip()
    return data

# Define regions of interest (ROIs) based on the form
regions_of_interest = {
    "Service Address": (10, 800, 5000, 200),
    "Year Established": (120, 420, 300, 50),
    "Material Used": (80, 600, 600, 50),
    "Replacement Year": (80, 850, 600, 50),
    "Signature": (100, 1200, 600, 50),
    "Phone Number": (500, 1350, 300, 50),
}

# Process the image
processed_image = preprocess_image(image_path)

# Extract text from the specified regions
extracted_data = extract_text_from_image(processed_image, regions_of_interest)

# Ensure output directory exists
output_dir = 'Data'
os.makedirs(output_dir, exist_ok=True)

# Save the extracted data into a spreadsheet
output_path = os.path.join(output_dir, 'extracted_data.xlsx')
df = pd.DataFrame([extracted_data])
df.to_excel(output_path, index=False)

print(f"Data extracted and saved to {output_path}")
