import Constant
from collections import Counter


def count_occurrences_in_file(filename, value):
    """Count the number of lines in which a value appears in a file."""
    with open(filename, 'r') as file:
        lines = file.readlines()
        return sum(1 for line in lines if str(value) in line)


def Comparing_write_breaks():
    # Paths to the files
    star_login = Constant.SERVER_PATH + "\star_login.csv"
    count_star_login = count_occurrences_in_file(star_login, 9999)

    # List to store the counts for each star_reg file
    count_star_regs = []

    # Loop through all star files
    for i in range(10):
        star_reg = Constant.SERVER_PATH + f"\star_{i}.csv"
        count_star_regs.append(count_occurrences_in_file(star_reg, 9999))

        # Find the most repeated value in count_star_regs
    counter = Counter(count_star_regs)
    most_repeated_value = counter.most_common(1)[0][0]
    print(most_repeated_value)
    # Compare counts
    if count_star_login == most_repeated_value:
        return True
    else:
        return False


#print(Comparing_write_breaks())
