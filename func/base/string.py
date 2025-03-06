#### extract smallest and largest floats in a string
import re

def extract_smallest_largest_floats(input_string):
    floats = re.findall(r"[-+]?\d*\.\d+", input_string)
    if not floats:
        return None, None
    floats = [float(num) for num in floats]
    smallest = round(min(floats), 2)
    largest = round(max(floats), 2)
    return smallest, largest

#### extract numbers from a string
import re

def extract_frame_number(file_name):
    # Extract the frame number from the image file name, frame_00001.jpg -> 1
    return int(re.search(r'\d+', file_name).group())

#### extract alphabets from a string
def extract_alpha(file_name):
    return ''.join(filter(str.isalpha, file_name))