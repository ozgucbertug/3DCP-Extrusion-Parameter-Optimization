import pandas as pd
import re

def extract_numeric(value):
    """
    Extracts the first numeric part of a string and converts it to a float.

    Parameters:
    value (str or float): The value to process

    Returns:
    float: The extracted numeric value or the original float
    """
    if isinstance(value, str):
        match = re.search(r"(\d+(\.\d+)?)", value)
        return float(match.group()) if match else None
    return value

# Path to the original data file
original_file_path = 'd:/GitHub/Extrusion-Parameter-Optimization/data.csv'

# Path to save the cleaned data
cleaned_file_path = 'd:/GitHub/Extrusion-Parameter-Optimization/data_clean.csv'

# Load the data
data = pd.read_csv(original_file_path)

# Drop unnecessary 'Unnamed' columns and the 'Pressure' column
data.drop(data.columns[data.columns.str.contains('Unnamed')], axis=1, inplace=True)
data.drop('Pressure', axis=1, inplace=True)

# Handle missing values in 'RunOrder' by forward fill
data['RunOrder'] = data['RunOrder'].ffill()

# Clean and convert all columns to numeric, handling special cases
for column in data.columns:
    data[column] = data[column].apply(extract_numeric)

# Group by 'RunOrder' and impute missing values in 'Layer Height' and 'Layer Width' using median within each group
for column in ['Layer Height', 'Layer Width']:
    data[column] = data.groupby('RunOrder')[column].transform(lambda x: x.fillna(x.median()))

# Dropping rows with missing or non-numeric values
data.dropna(inplace=True)

# Save the cleaned data
data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to {cleaned_file_path}")
