import GAN_pythorch_prediction
import math

def set_threshold_based_on_magnitude(value):
    """Set the STD_DEV_THRESHOLD based on the magnitude of the given value."""
    magnitude = math.floor(math.log10(abs(value)))
    return 10 ** magnitude

def read_GAN_score_from_file(file_path):
    """Read the GAN score from the given file."""
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            return float(content)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except ValueError:
        print("The content of the file is not a valid numerical value.")
        return None

def write_prediction_to_file(file_path, prediction):
    """Write the prediction outcome to the given file."""
    try:
        with open(file_path, 'w') as file:
            file.write(str(prediction))
    except Exception as e:
        print(f"Error writing to file: {e}")

def main():
    score_file_path = "D:\\works\\SignProj\\GAN-Model\\CudaProject\\conf.txt"
    result_file_path = r"D:\works\SignProj\GAN-Model\CudaProject\result.txt"

    # Read GAN score
    curr_GAN_score = read_GAN_score_from_file(score_file_path)
    if curr_GAN_score is None:
        return  # Exit if there was an error reading the score

    print(f"Read value: {curr_GAN_score}")

    # Dynamically set the threshold based on the magnitude of the read value
    STD_DEV_THRESHOLD = set_threshold_based_on_magnitude(curr_GAN_score)

    # Make prediction based on the score
    prediction_outcome = GAN_pythorch_prediction.predict_login(curr_GAN_score, STD_DEV_THRESHOLD)
    print(prediction_outcome)

    # Write prediction outcome to file
    write_prediction_to_file(result_file_path, prediction_outcome)

if __name__ == "__main__":
    main()
