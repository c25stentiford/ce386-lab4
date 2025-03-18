import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is the default camera index

# Capture a frame
ret, frame = cap.read()

# Release the camera
cap.release()

# Only process of ret is True
if ret:
    # Convert BGR (OpenCV default) to RGB for TFLite
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to a NumPy array
    img_array = np.array(frame_rgb, dtype=np.uint8)
    print("Image shape:", img_array.shape)  # Ensure shape matches model input

    # Preview the image
    cv2.imshow("Captured Image", frame)
    print("Press any key to exit.")
    while True:
        # Window stays open until key press
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
            break

else:
    print("Failed to capture image.")