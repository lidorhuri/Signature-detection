import cv2
import os

def ImageSharpingFunc():
    input_directory = r"D:\works\SignProj\GAN-Model\CudaProject\GAN_generated_samples"
    output_directory = r"D:\works\SignProj\GAN-Model\CudaProject\cleaned_samples"

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    for i in range(10):
        image_path = os.path.join(input_directory, f"generated_sample_{i}.png")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply thresholding
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove noise using opening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Blend the cleaned image with the original image to make the black less aggressive
        alpha = 0.7  # Weight for the original image (0 <= alpha <= 1)
        blended = cv2.addWeighted(img, alpha, cleaned, 1 - alpha, 0)

        output_path = os.path.join(output_directory, f"cleaned_sample_{i}.png")
        cv2.imwrite(output_path, blended)

    print("Image processing completed.")

