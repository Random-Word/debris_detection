"""
Convert XML specification files to text files for YOLOv8 training.
"""
import xml.etree.ElementTree as ET
import os
import argparse
import math

def convert_xml_to_txt(xml_dir, txt_dir):
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    total_files = len(xml_files)
    ten_percent = math.ceil(total_files / 10)

    # Iterate over each XML file in the directory
    for i, filename in enumerate(xml_files, start=1):
        if filename.endswith('.xml'):
            # Parse the XML file
            tree = ET.parse(os.path.join(xml_dir, filename))
            root = tree.getroot()

            # Get the image size
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)

            # List to hold the bounding box information
            boxes = []

            # Iterate over each 'object' element in the XML
            for obj in root.iter('object'):
                # Get the bounding box coordinates
                xmin = float(obj.find('bndbox/xmin').text)
                ymin = float(obj.find('bndbox/ymin').text)
                xmax = float(obj.find('bndbox/xmax').text)
                ymax = float(obj.find('bndbox/ymax').text)

                # Convert the coordinates to relative values
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                # Add the bounding box information to the list
                boxes.append(f"0 {x_center} {y_center} {box_width} {box_height}")

            # Write the bounding box information to a text file
            with open(os.path.join(txt_dir, filename.replace('.xml', '.txt')), 'w') as f:
                f.write('\n'.join(boxes))

            # Print progress every 10%
            if i % ten_percent == 0:
                print(f"Processed {i} out of {total_files} files ({(i/total_files)*100}%).")


# Parse command line arguments
parser = argparse.ArgumentParser(description='Convert XML files to text files.')
parser.add_argument('xml_dir', help='Directory containing the XML files.')
parser.add_argument('txt_dir', help='Directory to save the text files.')
args = parser.parse_args()

convert_xml_to_txt(args.xml_dir, args.txt_dir)