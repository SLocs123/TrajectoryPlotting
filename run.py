import pandas as pd
import re
import ast

def read_polygons_from_csv(file_path):
    polygons = []

    with open(file_path, 'r') as f:
        for line in f:
            # Strip any leading/trailing whitespace or newline characters
            line = line.strip()
            
            # Skip empty lines if any
            if not line:
                continue
            
            # Use regex to extract the polygon coordinates
            match = re.match(r'^Polygon \d+: (.+)$', line)
            if match:
                polygon_data_str = match.group(1)
                
                # Convert the extracted string to a Python dictionary
                try:
                    polygon_data = ast.literal_eval(polygon_data_str)
                    polygons.append(polygon_data)
                except ValueError as e:
                    print(f"Error parsing line: {line}. Error: {e}")

    # Create a DataFrame from the list of polygons
    df = pd.DataFrame(polygons)
    return df


poly = read_polygons_from_csv('Filtered_Detection_Zone_Annotations.csv')
poly.to_csv('polygons.csv', index=False)
