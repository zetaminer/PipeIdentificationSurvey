import pytesseract
import cv2
import numpy as np
import pandas as pd
import os

# Specify Tesseract OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Preprocessing functions with previews for the "Service Address" section

def dynamic_brightness_contrast(image, roi):
    """
    Allows dynamic adjustment of brightness and contrast using trackbars.
    """
    adjusted_image = image.copy()


    def update_brightness_contrast(_=None):
        """
        Update the brightness and contrast dynamically based on trackbar values.
        """
        try:
            # Safely get the positions of the trackbars

            alpha = cv2.getTrackbarPos('Contrast', 'Brightness & Contrast') / 10  # Scale for contrast
            beta = cv2.getTrackbarPos('Brightness', 'Brightness & Contrast') - 100  # Scale for brightness

            # Apply brightness and contrast adjustments
            nonlocal adjusted_image
            adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            # Crop the ROI for display
            x, y, w, h = roi
            cropped = adjusted_image[y:y+h, x:x+w]

            # Display the adjusted ROI
            cv2.imshow('Brightness & Contrast', cropped)
        except cv2.error as e:
            print("OpenCV error occurred:", e)
            return

    # Create the window for display
    cv2.namedWindow('Brightness & Contrast', cv2.WINDOW_AUTOSIZE)


    # Create trackbars for contrast and brightness
    cv2.createTrackbar('Contrast', 'Brightness & Contrast', 10, 30, update_brightness_contrast)  # Default = 1.0
    cv2.createTrackbar('Brightness', 'Brightness & Contrast', 100, 200, update_brightness_contrast)  # Default = 0

    # Trigger the first update manually to initialize the display
    update_brightness_contrast()

    # Wait until the user presses a key to finalize adjustments
    while True:
        key = cv2.waitKey(1)
        if key == 27:  # Escape key
            break

    # Close the window
    cv2.destroyAllWindows()

    return adjusted_image




def adjust_brightness_contrast(image, alpha=2.0, beta=50, roi=None):
    """
    Adjusts the brightness and contrast of an image.
    """
    # adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # if roi:
    #     preview_roi(adjusted, roi, "Brightness & Contrast Adjustment")
    # return adjusted
    print("Adjusting Brightness and Contrast")
    dynamic_brightness_contrast(img, service_address_roi)


def apply_clahe(image, roi=None):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    if roi:
        preview_roi(enhanced, roi, "CLAHE (Contrast Enhancement)")
    return enhanced

def sharpen_image(image, roi=None):
    """
    Applies a sharpening filter to the image.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    if roi:
        preview_roi(sharpened, roi, "Sharpened Image")
    return sharpened

def adaptive_threshold(image, roi=None):
    """
    Applies adaptive thresholding to binarize the image.
    """
    thresholded = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    if roi:
        preview_roi(thresholded, roi, "Adaptive Thresholding")
    return thresholded

def denoise_image(image, roi=None):
    """
    Denoises the image to remove small artifacts.
    """
    denoised = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    if roi:
        preview_roi(denoised, roi, "Denoised Image")
    return denoised

def preview_roi(image, roi, title):
    """
    Displays a specific ROI (Region of Interest) for preview.
    """
    x, y, w, h = roi
    cropped = image[y:y+h, x:x+w]
    cv2.imshow(title, cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(image_path, service_address_roi):
    """
    Preprocesses the image using multiple steps to improve OCR readability,
    previewing only the "Service Address" section.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print("Displaying Original Image for Service Address ROI")
    preview_roi(img, service_address_roi, "Original Image - Service Address")

    # Dynamically adjust brightness and contrast
    print("Adjusting Brightness and Contrast")
    dynamic_brightness_contrast(img, service_address_roi)

    # Apply CLAHE
    img = apply_clahe(img, roi=service_address_roi)

    # Sharpen the image
    img = sharpen_image(img, roi=service_address_roi)

    # Adaptive thresholding
    img = adaptive_threshold(img, roi=service_address_roi)

    # Optional: Denoise the image
    img = denoise_image(img, roi=service_address_roi)

    return img


# Perform OCR on specific regions
def extract_text_from_image(image, regions):
    """
    Extracts text from specific regions of an image using OCR.
    """
    data = {}
    for field, roi in regions.items():
        x, y, w, h = roi
        cropped = image[y:y+h, x:x+w]
        # OCR
        text = pytesseract.image_to_string(cropped, config='--psm 6')
        data[field] = text.strip()
    return data

# Main execution
if __name__ == "__main__":
    # Path to the image
    image_path = r'C:\Users\julia\PycharmProjects\IvanPipeProject\Scans\Scan2024-12-19_095936.jpg'

    # Define regions of interest (ROIs) based on the form
    regions_of_interest = {
        "Service Address": (522, 533, 1728, 131),  # Coordinates for "Service Address"
        "Year Established": (120, 420, 300, 50),
        "Material Used": (80, 600, 600, 50),
        "Replacement Year": (80, 850, 600, 50),
        "Signature": (100, 1200, 600, 50),
        "Phone Number": (500, 1350, 300, 50),
    }

    # Preprocess the image
    service_address_roi = regions_of_interest["Service Address"]
    processed_image = preprocess_image(image_path, service_address_roi)



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
