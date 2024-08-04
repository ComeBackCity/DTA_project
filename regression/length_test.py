import pandas as pd

def analyze_string_lengths(file_path, column_name):
    # Load the DataFrame
    df = pd.read_csv(file_path, sep="\t")

    # Ensure the column exists and contains strings
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Calculate the length of each string in the specified column
    string_lengths = df[column_name].drop_duplicates().astype(str).apply(len)

    # Calculate the average length of the strings
    average_length = string_lengths.mean()

    # Count the number of strings with length greater than 100
    count_greater_than_100 = (string_lengths > 2000).sum()

    return average_length, count_greater_than_100

# Usage example
file_path = './data/kiba.tab'
column_name = 'X2'
average_length, count_greater_than_100 = analyze_string_lengths(file_path, column_name)

print(f"Average length of strings in column '{column_name}': {average_length}")
print(f"Number of strings with length greater than 100: {count_greater_than_100}")
