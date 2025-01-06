import cv2
import numpy as np

# Create a blank image
img = np.zeros((300, 300, 3), dtype=np.uint8)

# Draw a rectangle on the image
cv2.rectangle(img, (50, 50), (250, 250), (0, 255, 0), -1)

# Display the image
cv2.imshow('Test Window', img)



# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

