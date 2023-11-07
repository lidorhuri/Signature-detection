import matplotlib.pyplot as plt
import numpy as np
import os
import Constant
import Center_Points

def Save_registred_signatures_image():
    # Create the output folder if it doesn't exist
    os.makedirs(Constant.OUTPUT_IMAGE_FOLDER, exist_ok=True)

    for i in range(10):
        # Create a new figure for the current signature
        fig, ax = plt.subplots(figsize=(8, 8))

        # Get the CSV file path for the current signature
        csv_path = f"{Constant.SERVER_PATH}\\star_{i}.csv"

        # Read the CSV file into a pandas DataFrame
        curr_signature = np.loadtxt(csv_path, delimiter=',')

        # Extract X and Y columns
        x = curr_signature[:, 0]
        y = curr_signature[:, 1]



        curr_signature = Center_Points.Center_Array_Point_On_Board([x, y])

        # Extract X and Y columns after the fix through center
        x = curr_signature[:, 0]
        y = curr_signature[:, 1]

        # Initialize a variable to store the previous point
        prev_point = None

        # Iterate over points and plot lines
        for j in range(len(x)):
            # Check if the current point is (9999, 9999)
            if x[j] >= 2000 and y[j] >= 2000:
                prev_point = None  # Reset the previous point and move to the next point
                continue

            # Check if there's a previous point
            if prev_point is not None:
                ax.plot([prev_point[0], x[j]], [prev_point[1], y[j]], linestyle='-', color='black')

            prev_point = [x[j], y[j]]

        # Set labels and title for the current subplot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Set X and Y axis limits
        ax.set_xlim(0, Constant.SIGNATURE_SCREEN_WIDTH)
        ax.set_ylim(0, Constant.SIGNATURE_SCREEN_WIDTH)#150
        ax.invert_yaxis()

        # Hide the X and Y axes elements
        ax.axis('off')

        # Save the current signature as an image (PNG format)
        output_path = os.path.join(Constant.OUTPUT_IMAGE_FOLDER, f"{i}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    print("All signatures saved as individual images.")



def Save_login_signture_image():
    # Create the output folder if it doesn't exist

    os.makedirs(Constant.OUTPUT_IMAGE_FOLDER_LOGIN, exist_ok=True)

    # Create a new figure for the login.csv signature
    fig_login, ax_login = plt.subplots(figsize=(8, 8))
    login_csv_path = f"{Constant.SERVER_PATH}\\star_login.csv"

    # Read the CSV file into a pandas DataFrame
    login_signature = np.loadtxt(login_csv_path, delimiter=',')
    x_login = login_signature[:, 0]
    y_login = login_signature[:, 1]

    login_signature = Center_Points.Center_Array_Point_On_Board([x_login, y_login])

    # extract the x and y again after center fix
    x_login = login_signature[:, 0]
    y_login = login_signature[:, 1]

    # Initialize a variable to store the previous point
    prev_point = None

    # Iterate over points and plot lines
    for j in range(len(x_login)):
        # Check if the current point is (9999, 9999)
        if x_login[j] >= 2000 and y_login[j] >= 2000:
            prev_point = None  # Reset the previous point and move to the next point
            continue

        # Check if there's a previous point
        if prev_point is not None:
            ax_login.plot([prev_point[0], x_login[j]], [prev_point[1], y_login[j]], linestyle='-', color='black')

        prev_point = [x_login[j], y_login[j]]

    # Set labels and title for the login.csv subplot
    ax_login.set_xlabel('X')
    ax_login.set_ylabel('Y')

    # Set X and Y axis limits for the login.csv subplot
    ax_login.set_xlim(0, Constant.SIGNATURE_SCREEN_WIDTH)
    ax_login.set_ylim(0, Constant.SIGNATURE_SCREEN_WIDTH)#300
    ax_login.invert_yaxis()

    # Hide the X and Y axes elements for the login.csv subplot
    ax_login.axis('off')

    # Save the login.csv signature as an image (PNG format)
    output_path_login = os.path.join(Constant.OUTPUT_IMAGE_FOLDER_LOGIN, "login.png")
    plt.savefig(output_path_login, bbox_inches='tight', pad_inches=0)
    plt.close(fig_login)

    print("Login signature saved as 'login.png'.")

# save images for next stage at GAN!



# Example usage:
#Save_registred_signatures_image()
#Save_login_signture_image()