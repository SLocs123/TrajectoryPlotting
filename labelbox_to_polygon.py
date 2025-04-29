import json
import csv
import time

read = json.load(open('cam04.ndjson'))
polygons = []
for label_id, label_data in read['projects'].items():
    # Iterate over annotations in the label
    for annotation in label_data['labels'][0]['annotations']['objects']:
        if annotation['annotation_kind'] == 'ImagePolygon':
            polygon_coords = annotation['polygon']
            polygons.append(polygon_coords)


filename = 'cam04_poly.csv'
max_points = max(len(poly) for poly in polygons)
# Open the file in write mode ('w', newline='') to avoid extra newline characters
with open(filename, mode='w', newline='') as file:
    # Create a CSV writer object
    csv_writer = csv.writer(file)
    # Write the header row
    header = [i for i in range(max_points)]
    csv_writer.writerow(header)
    
    # Write each polygon as a row in the CSV file
    for polygon in polygons:
        # Fill missing points with empty strings to ensure consistent row length
        row = []
        for idx, _ in enumerate(polygon):
            item = polygon[idx]
            x = item['x']
            y = item['y']
            row.append(str([x,y]))
        while len(row) < max_points:
            row.append('')
        csv_writer.writerow(row)

print(f'CSV file "{filename}" has been written successfully.')