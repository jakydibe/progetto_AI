import cv2
import numpy as np

# Step 1: Read the Image
image = cv2.imread('C:\\Users\\jakyd\\Desktop\\progetto_AI\\first_cell.png')

# Step 2: Preprocess the Image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Step 3: Find Contours (assuming characters are enclosed in cells)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store coordinates of grid cells
cells = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cells.append((x, y, w, h))
    # Optionally, draw the contours for visualization
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Step 4: Sort the cell coordinates if necessary (e.g., left-to-right, top-to-bottom)
cells = sorted(cells, key=lambda b: (b[1], b[0]))  # Sort by y, then by x

# Step 5: Extract and Save Sub-images
for idx, (x, y, w, h) in enumerate(cells):
    sub_image = image[y:y+h, x:x+w]
    cv2.imwrite(f'C:\\Users\\jakyd\\Desktop\\progetto_AI\\sub_image_{idx}.jpg', sub_image)

# Optionally, show the original image with drawn contours
cv2.imshow('Image with Grid', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
