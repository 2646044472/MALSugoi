import pandas as pd
import glob
import re

# Define the range of ranks to merge
start_rank = 3000
end_rank = 4000

# Set file path and pattern, read all CSV files from the data/anime_info directory
file_paths = glob.glob("data/anime_info/anime_data_*.csv")

# List to store dataframes from each file
dfs = []

# Regular expression to extract rank range from filenames
rank_pattern = re.compile(r"anime_data_(\d+)_to_(\d+)\.csv")

# Loop through each file and filter based on rank range
for file in file_paths:
    match = rank_pattern.search(file)
    if match:
        file_start_rank = int(match.group(1))
        file_end_rank = int(match.group(2))

        # Check if file's rank range falls within the specified range
        if file_start_rank >= start_rank and file_end_rank <= end_rank:
            df = pd.read_csv(file)
            dfs.append(df)

# Check if any matching files were found
if dfs:
    # Merge all dataframes into one
    merged_df = pd.concat(dfs, ignore_index=True)

    # Keep only the specified columns
    columns_to_keep = ['title', 'score', 'genres', 'ranked', 'popularity', 'members', 'favorites']
    filtered_df = merged_df[columns_to_keep]

    # Save the merged data to a new CSV file
    output_file = f"data/anime_info/anime_data_{start_rank}_to_{end_rank}.csv"
    filtered_df.to_csv(output_file, index=False)

    print(f"Data has been saved to: {output_file}")
else:
    print(f"No files found with ranks between {start_rank} and {end_rank}.")