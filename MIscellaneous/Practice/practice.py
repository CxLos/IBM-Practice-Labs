# Imports
import pandas as pd 

# File path
message_file = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\Miscellaneous\Practice\msg_file.txt'
message_file1 = r'C:\Users\CxLos\OneDrive\Documents\IBM Data Analyst Professional Certificate\IBM Practice Labs\Miscellaneous\Practice\coding_qual_input.txt'

# read contents of .txt file
with open(message_file, 'r') as file:
    content = file.read()

# print(content)
    
# Function to get "I love Computers" 
def read_message(file_path):
    
    # Read the text file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract words for positions 1, 3, and 6
    positions_to_extract = [1, 3, 6]

    # splits line into a list of words based on whitespace
    words_and_positions = [(int(line.split()[0]), # converts first element of the split line into an integer
                            line.split()[1]) # retrieves second element of the split line.
                            for line in lines if int(line.split()[0]) in positions_to_extract]

    # Sort the words based on their positions
    sorted_words = sorted(words_and_positions, key=lambda x: x[0])

    # Extract the words only
    words = [word for _, word in sorted_words]

    # Create the final message
    message = ' '.join(words)

    return message

result = read_message(message_file1)
print(result)