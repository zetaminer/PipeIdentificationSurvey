import cv2

# Load the image
image_path = r'C:\Users\julia\PycharmProjects\IvanPipeProject\Scans\Scan2024-12-19_095936.jpg'
img = cv2.imread(image_path)

# Clone the image to allow dynamic updating
temp_img = img.copy()

# Variables to store the starting and ending coordinates of the rectangle
start_point = None
end_point = None
drawing = False  # Flag to indicate when the user is drawing

# Function to capture mouse events
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, temp_img

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing when left mouse button is pressed
        drawing = True
        start_point = (x, y)
        print(f"Start Point: {start_point}")

    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Update rectangle while dragging
        temp_img = img.copy()  # Reset to the original image
        end_point = (x, y)
        cv2.rectangle(temp_img, start_point, end_point, (0, 255, 0), 2)  # Draw rectangle
        cv2.imshow('Image', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:  # Finalize the rectangle when left mouse button is released
        drawing = False
        end_point = (x, y)
        cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)  # Draw final rectangle
        cv2.imshow('Image', img)
        print(f"Rectangle: Start Point={start_point}, End Point={end_point}, Width={end_point[0] - start_point[0]}, Height={end_point[1] - start_point[1]}")

# Display the image and set up the mouse callback
cv2.imshow('Image', img)
cv2.setMouseCallback('Image', draw_rectangle)

# Wait for the user to press a key
cv2.waitKey(0)
cv2.destroyAllWindows()
