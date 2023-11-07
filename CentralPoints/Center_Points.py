import numpy as np
import matplotlib.pyplot as plt
import Constant

def Center_Array_Point_On_Board(arr):  # arr = [arr_x ,arr_y] , returns the arr after shift to the center

    # Initialize arrays to store adjusted data
    all_x_values = []
    all_y_values = []

    # Extract X and Y columns
    temp_x = arr[0]
    temp_y = arr[1]

    # calculate without 9999
    x_values = [temp_x[i] for i in range(len(temp_x)) if temp_x[i] != 9999]
    y_values = [temp_y[i] for i in range(len(temp_y)) if temp_y[i] != 9999]

    # Calculate average X and Y values (min and max)
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    diff_x = max_x - min_x
    diff_y = max_y - min_y

    ##the center point!!
    average_x = min_x + (diff_x/2)
    average_y = min_y + (diff_y/2)
    ##

    # travel distance to the central point of the page
    travel_x = Constant.CENTER_X_POINT - average_x
    travel_y = Constant.CENTER_Y_POINT - average_y

    # Adjust points using differences
    temp_x += travel_x
    temp_y += travel_y

    # Append adjusted points to the arrays
    all_x_values.extend(temp_x)
    all_y_values.extend(temp_y)

    # Store adjusted points as a stroke
    stroke = np.column_stack((temp_x, temp_y))

    return stroke


def Center_Register_Signatures():  # triggers the Registered signature in the SERVER
    all_x_values = []
    all_y_values = []
    all_shifted_signatures = []
    # Path to the directory containing CSV files
    # SERVER_PATH

    # Extract X and Y columns from each CSV file and adjust the points
    for i in range(10):
        csv_path = f"{Constant.SERVER_PATH}\\{i}.csv"
        curr_signature = np.loadtxt(csv_path, delimiter=',')

        # Extract X and Y columns
        x_values = curr_signature[:, 0]
        y_values = curr_signature[:, 1]

        shifted_arr = Center_Array_Point_On_Board([x_values, y_values])
        #print(shifted_arr)
        all_x_values.append(x_values)
        all_y_values.append(y_values)
        all_shifted_signatures.append(shifted_arr)  # for plotting only!!

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Iterate through each dataset and plot its points
    cnt = 1
    for dataset in all_shifted_signatures:
        x_values = [point[0] for point in dataset]
        y_values = [point[1] for point in dataset]
        ax.scatter(x_values, y_values)
        cnt += 1

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of X vs Y from All CSV Files')
    plt.grid()
    plt.xlim(0, Constant.SIGNATURE_SCREEN_WIDTH)
    plt.ylim(0, Constant.SIGNATURE_SCREEN_WIDTH)#150
    plt.gca().invert_yaxis()

    # Show the plot
    #plt.show()

    ### fixing x and y values

    all_x_values = np.concatenate(all_x_values)
    # Flatten the table to a 1D array
    flattened_array = all_x_values.flatten()
    # Reshape the flattened array to a single column
    all_x_values = flattened_array.reshape(-1, 1)

    all_y_values = np.concatenate(all_y_values)
    # Flatten the table to a 1D array
    flattened_array = all_y_values.flatten()
    # Reshape the flattened array to a single column
    all_y_values = flattened_array.reshape(-1, 1)

    return [all_x_values, all_y_values]






###########Script_test###########
#Center_Register_Signatures()
#################################