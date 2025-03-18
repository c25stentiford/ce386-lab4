"""This script loads a .tflite model into LiteRT and continuously takes pictures with a webcam,
printing if the picture is of a cat or a dog."""

import cv2
from ai_edge_litert.interpreter import Interpreter, SignatureRunner
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


# Convert picture to numpy for model ingest
def convert(frame) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.array(frame_rgb, dtype=np.uint8)


def main():

    # Verify arguments and print usage if not correct
    if len(sys.argv) != 2:
        print("Usage: python " + sys.argv[0] + " <model_path.tflite>")
        exit(1)

    # Create LiteRT SignatureRunner from model path given as argument
    model_path = sys.argv[1]
    runner = get_litert_runner(model_path)

    # Init webcam
    webcam = cv2.VideoCapture(0)  # 0 is default camera index

    while True:
        ret, frame = webcam.read()
        if ret:
            # convert image
            img_array = convert(frame)
            # read image into pillow and resize
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

    # Release the camera
    webcam.release()
    print("Program complete")


# Executes when script is called by name
if __name__ == "__main__":
    main()
