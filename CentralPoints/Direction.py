import csv
import math
import os
import Constant

# Function to calculate the angle between two vectors
def calculate_angle(x1, y1, x2, y2):
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg = math.degrees(angle_rad) - 90

    if angle_deg < 0:
        angle_deg += 360  # Convert negative angles to positive angles

    return int(angle_deg)


# Determine the movement direction based on the angle
def determine_movement_direction(angle):
    if (angle >= 0 and angle <= 10) or (angle >= 350 and angle <= 360):
        return "UP"
    elif angle >= 11 and angle <= 80:
        return "UP_RIGHT"
    elif angle >= 81 and angle <= 100:
        return "RIGHT"
    elif angle >= 101 and angle <= 170:
        return "DOWN_RIGHT"
    elif angle >= 171 and angle <= 190:
        return "DOWN"
    elif angle >= 191 and angle <= 260:
        return "DOWN_LEFT"
    elif angle >= 261 and angle <= 280:
        return "LEFT"
    elif angle >= 281 and angle <= 349:
        return "UP_LEFT"


def main_movement_function(filename):
    # Read login signature
    login_signature = []
    movment_changed = []

    file_path = os.path.join(Constant.SERVER_PATH, f'{filename}.csv')
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        prev_point = None
        login_stroke_direction_vectors = []
        for row in csvreader:
            if prev_point == None:
                prev_point = row
                continue

            angle = calculate_angle(int(prev_point[0]), int(prev_point[1]), int(row[0]), int(row[1]))
            login_stroke_direction_vectors.append({'angle': angle})
            prev_point = row
            direction = determine_movement_direction(angle)
            movment_changed.append(direction)

    movment_changed.reverse()
    # print(movment_changed)  # Print the list of movement changes  print('Login signature rejected.')
    return movment_changed


def find_similar_array(login_array, registered_arrays):
    for registered_array in registered_arrays:
        if compare_arrays(login_array, registered_array):
            return True
    return False

def compare_arrays(array1, array2):
    deviations = 0
    i_1 = 0
    i_2 = 0
    stopFlag = False

    while stopFlag == 0:
        # 3 options : 1) both are done
        #             2) one of them is in the next different element and one is done
        #             3) both are in the next different element

        ## 1)
        if i_1 == (len(array1) - 1) and i_2 == (len(array2) - 1):
            stopFlag = True
            continue
        ## 2)
        elif i_1 == (len(array1) - 1) or i_2 == (len(array2) - 1):
            if i_1 == (len(array1) - 1):
                # counts the changes is movements
                i_2 += 1  # get to the next element to check
                while i_2 <= len(array2) - 1:
                    if array2[i_2] != array2[i_2 - 1]:
                        deviations += 1
                    i_2 += 1
            else:
                # counts the changes is movements
                i_1 += 1  # get to the next element to check
                while i_1 <= len(array1) - 1:
                    if array1[i_1] != array1[i_1 - 1]:
                        deviations += 1
                    i_1 += 1
            stopFlag = True
            continue

        ## 3)
        direc_1 = array1[i_1]
        direc_2 = array2[i_2]
        # if the 2 elements are equal
        if direc_2 == direc_1:
            # get the 2 arrays to the next different element to compare
            while direc_1 == array1[i_1] and i_1 < (len(array1) - 1):
                i_1 += 1
            while direc_2 == array2[i_2] and i_2 < (len(array2) - 1):
                i_2 += 1
            continue
        else:
            deviations += 1
            i_1 += 1
            i_2 += 1
            continue

    th = max(len(array1), len(array2)) * 0.15
    # acc = 85%
    if deviations >= th:
        return False  # Too many deviations, not similar

    return True  # If deviations are within allowed limits, arrays are similar


def activate_direction():
    filename_to_process = "login"
    movment_changed_login = main_movement_function(filename_to_process)

    movment_changed_registered = []
    for i in range(0, 10):
        filename = str(i)
        movment_changed_registered.append(main_movement_function(filename))

        # Assuming you have movment_changed_login and movment_changed_registered from your code
    similar = find_similar_array(movment_changed_login, movment_changed_registered)

    if similar:
        print("There is similarity in the directions.")
        return True
    else:
        print("There is no similarity in the directions.")
        return False

###########Script_test###########
#activate_direction()
#################################