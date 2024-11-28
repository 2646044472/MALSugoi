import pandas as pd

# Function to convert CSV to text format
def csv_to_text(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Initialize an empty list to hold the text lines
    text_lines = []

    # Iterate over each row and convert it to a text line
    for index, row in df.iterrows():
        # Directly append each row as a string to the text_lines list
        text_lines.append(','.join(map(str, row.values)))

    # Join the lines into a single string with newlines between each
    text = "\n".join(text_lines)
    return text

# Example usage: Convert the CSV file to text
csv_file_path = r'C:/Users/Lenovo/OneDrive/文档/GitHub/MALSugoi/anime_info.csv'  # Adjust the file path
text_data = csv_to_text(csv_file_path)

# Print the resulting text data
print(text_data)

