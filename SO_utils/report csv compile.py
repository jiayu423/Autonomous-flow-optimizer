import os
import pandas as pd

def search_and_compile_csv(input_folder, output_file, csv_filename):
    # List to store data frames from CSV files
    df_list = []

    # Iterate through all directories and subdirectories
    for root, dirs, files in os.walk(input_folder):
        # Check if the CSV file exists in the current directory
        if csv_filename in files:
            csv_path = os.path.join(root, csv_filename)
            # Read CSV file and append its content to df_list
            df = pd.read_csv(csv_path)
            df_list.append(df)

    # Concatenate all data frames in df_list
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        # Write the concatenated data frame to a single CSV file
        final_df.to_csv(output_file, index=False)
        print(f"CSV files compiled successfully into {output_file}")
    else:
        print("No CSV files found.")

# Specify the input folder, output file, and CSV filename to search for
input_folder = '/path/to/your/input/folder'
output_file = '/path/to/your/output/compiled.csv'
csv_filename = 'specific_filename.csv'

# Call the function to search and compile CSV files
search_and_compile_csv(input_folder, output_file, csv_filename)