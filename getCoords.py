import cv2

# Load the image
image_path = r'C:\Users\julia\PycharmProjects\IvanPipeProject\Scans\Scan2024-12-19_095936.jpg'
img = cv2.imread(image_path)

# Function to capture mouse clicks and print coordinates
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Coordinates: x={x}, y={y}")
        # Draw a small circle at the point clicked
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Image', img)

# Display the image and set up the mouse callback
cv2.imshow('Image', img)
cv2.setMouseCallback('Image', get_coordinates)

# Draw rectangles on the original image for each ROI
cv2.rectangle(img, (50, 200), (50+600, 200+50), (0, 255, 0), 2)  # Service Address
cv2.rectangle(img, (100, 400), (100+200, 400+50), (0, 255, 0), 2)  # Year

# Show the image with rectangles
cv2.imshow('Marked Image', img)

# Wait for the user to press a key
cv2.waitKey(0)
cv2.destroyAllWindows()
