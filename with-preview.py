import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import sys
import numpy as np
from PIL import Image


def get_litert_runner(model_path: str) -> SignatureRunner:
    """Opens a .tflite model from path and returns a LiteRT SignatureRunner that can be called for inference
    Args:
        model_path (str): Path to a .tflite model
    Returns:
        SignatureRunner: An AI-Edge LiteRT runner that can be invoked for inference."""

    interpreter = Interpreter(model_path=model_path)
    # Allocate the model in memory
    interpreter.allocate_tensors()
    print(f"Allocated LiteRT with signatures {interpreter.get_signature_list()}")

    # Create callable object that runs inference based on signatures
    # 'serving_default' is default... but in production should parse from signature
    return interpreter.get_signature_runner("serving_default")


# Executes when script is called by name
if __name__ == "__main__":

    # Verify arguments and print usage if not correct
    if len(sys.argv) != 2:
        print("Usage: python " + sys.argv[0] + " <model_path.tflite>")
        exit(1)

    # Create LiteRT SignatureRunner from model path given as argument
    model_path = sys.argv[1]
    runner = get_litert_runner(model_path)

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera index

    # Capture a frame
    ret, frame = cap.read()

    # Release the camera
    cap.release()

    # Only process if ret is True
    if ret:
        # Convert BGR (OpenCV default) to RGB for TFLite
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to a NumPy array
        img_array = np.array(frame_rgb, dtype=np.uint8)

        scaled = Image.fromarray(img_array).resize(
            (150, 150), resample=Image.Resampling.LANCZOS
        )
        # convert back into uint8 array
        input_data = np.array(scaled, dtype=np.uint8)
        # "It's not TensorFlow without expand_dims(tm)!"
        expanded = np.expand_dims(input_data, 0)
        # execute inference
        output = runner(catdog_input=expanded)
        # print result
        if output["output_0"][0] > 0:
            print("Dog")
        else:
            print("Cat")

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
