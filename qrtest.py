import cv2
import numpy as np
import os
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt


def process_qr_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect QR codes in the image
    decoded_objects = decode(gray)
    
    # Process each detected QR code
    qr_locations = []
    for obj in decoded_objects:
        # Get the coordinates of the QR code
        points = obj.polygon
        if len(points) == 4:  # Ensure we have 4 points
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            
            # Get the bounding box of the QR code
            x_min = min(pt[0] for pt in points)
            y_min = min(pt[1] for pt in points)
            x_max = max(pt[0] for pt in points)
            y_max = max(pt[1] for pt in points)
            qr_locations.append({
                'data': obj.data.decode('utf-8'),
                'bounding_box': (x_min, y_min, x_max, y_max)
            })
    
    # Display the image with detected QR codes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected QR Codes")
    plt.show()
    
    return qr_locations

def main():
    image_path = "C:\\Users\\smaniaci\\Desktop\\qrtestFolder\\qrcode.jpg" # Change to your image path
    qr_locations = process_qr_image(image_path)
    
    if not qr_locations:
        print("No QR codes detected.")
    else:
        for qr in qr_locations:
            print(f"QR Code Data: {qr['data']}")
            print(f"Bounding Box: {qr['bounding_box']}")

if __name__ == "__main__":
    main()
