import os

# Define the base directory where your files are located
# This assumes a structure like:
# /path/to/your/files/
# ├── all_combined_text.txt
# ├── json/
# │   └── combined_json_data.txt
# └── webpages/
#     └── combined_webpage_data.txt
# Replace 'path/to/your/files' with your actual directory path.
directory = 'C:/Users/Administrator/Downloads/all types of files read/all the combined files text'

# A dictionary mapping a descriptive name to the file path.
# This makes it clear which "folder" or category each file belongs to.
files_to_combine = {
    'All Combined Text Files': 'combined_all_text.txt',
    'CSV Data': 'combined_data.txt',
    'JSON Data': 'combined_json_data.txt',
    'Webpage Data': 'combined_webpage_data.txt'
}

# Name of the output file
output_file = 'combined_output.txt'

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Loop through the dictionary to get both the descriptive name and the file path
    for description, filename in files_to_combine.items():
        file_path = os.path.join(directory, filename)
        
        # Check if the file exists before trying to read it
        if os.path.exists(file_path):
            print(f"Combining {description} from {filename}...")
            
            # Write a clear header to the output file
            outfile.write(f"/n/n--- Content from {description} ({filename}) ---/n/n")
            
            try:
                # Open the input file in read mode
                with open(file_path, 'r', encoding='utf-8') as infile:
                    # Read the content and write it to the output file
                    outfile.write(infile.read())
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                outfile.write(f"--- ERROR: Could not read {filename}. Error: {e} ---/n")
        else:
            print(f"File not found: {filename}. Skipping...")
            outfile.write(f"/n/n--- File not found: {filename}. Skipping. ---/n/n")

print(f"/nAll specified files have been combined into {output_file}")