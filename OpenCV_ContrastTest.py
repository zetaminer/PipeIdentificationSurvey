import cv2
import numpy as np


def dynamic_brightness_contrast(image, roi):
    """
    Allows dynamic adjustment of brightness and contrast using trackbars.
    """
    adjusted_image = image.copy()

    # 1. Create the window for display (use "Test Window")
    cv2.namedWindow('Test Window', cv2.WINDOW_AUTOSIZE)

    # 2. Create trackbars *in* "Test Window" with dummy callbacks
    cv2.createTrackbar('Contrast', 'Test Window', 10, 30, lambda x: None)
    cv2.createTrackbar('Brightness', 'Test Window', 100, 200, lambda x: None)

    # 3. Define the function that updates brightness/contrast
    def update_brightness_contrast():
        alpha = cv2.getTrackbarPos('Contrast', 'Test Window') / 10
        beta = cv2.getTrackbarPos('Brightness', 'Test Window') - 100

        nonlocal adjusted_image
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Crop ROI or show the whole image — up to you
        x, y, w, h = roi
        cropped = adjusted_image[y:y + h, x:x + w]

        # 4. Display inside the same "Test Window"
        cv2.imshow('Test Window', cropped)

    # 5. Main loop
    while True:
        # Update the display continuously
        update_brightness_contrast()

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # Escape key
            break

    cv2.destroyAllWindows()
    return adjusted_image


# ----------------
# Testing the function
# ----------------
if __name__ == "__main__":
    # Create a blank test image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw a rectangle on the image (just to see something)
    cv2.rectangle(img, (50, 50), (250, 250), (0, 255, 0), -1)

    # Convert to grayscale if desired
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define a region of interest (ROI)
    roi = (50, 50, 200, 200)  # (x, y, width, height)

    # Call the dynamic brightness/contrast GUI
    adjusted_img = dynamic_brightness_contrast(img_gray, roi)

    # Optionally do something with adjusted_img afterward
