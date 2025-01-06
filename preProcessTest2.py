def dynamic_brightness_contrast(image, roi):
    """
    Allows dynamic adjustment of brightness and contrast using trackbars.
    """
    adjusted_image = image.copy()

    def update_brightness_contrast(_):
        # Get current positions of trackbars
        alpha = cv2.getTrackbarPos('Contrast', 'Brightness & Contrast') / 10  # Scale for contrast
        beta = cv2.getTrackbarPos('Brightness', 'Brightness & Contrast') - 100  # Scale for brightness
        # Apply brightness and contrast adjustments
        nonlocal adjusted_image
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        # Crop the ROI
        x, y, w, h = roi
        cropped = adjusted_image[y:y+h, x:x+w]
        # Display the adjusted ROI
        cv2.imshow('Brightness & Contrast', cropped)

    # Create the window before creating trackbars
    cv2.namedWindow('Brightness & Contrast')

    # Initialize trackbars for contrast and brightness
    cv2.createTrackbar('Contrast', 'Brightness & Contrast', 10, 30, update_brightness_contrast)  # Default = 1.0
    cv2.createTrackbar('Brightness', 'Brightness & Contrast', 100, 200, update_brightness_contrast)  # Default = 0

    # Trigger the first update manually to show the initial state
    update_brightness_contrast(0)

    # Wait until the user presses a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return adjusted_image


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
    img = dynamic_brightness_contrast(img, service_address_roi)

    # Apply CLAHE
    img = apply_clahe(img, roi=service_address_roi)

    # Sharpen the image
    img = sharpen_image(img, roi=service_address_roi)

    # Adaptive thresholding
    img = adaptive_threshold(img, roi=service_address_roi)

    # Optional: Denoise the image
    img = denoise_image(img, roi=service_address_roi)

    return img
