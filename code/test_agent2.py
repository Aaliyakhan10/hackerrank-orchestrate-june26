import sys
import os
sys.path.append(os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv('../.env')

import json
import csv
from image_analyzer import ImageAnalyzer

analyzer = ImageAnalyzer()

def load_dataset():
    data = []
    with open('../dataset/sample_claims.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

claims = load_dataset()
for row in claims:
    if row[0] in ['user_030', 'user_034']:
        print(f"\n--- Row ({row[0]}) ---")
        images = row[1].split(';')
        for img in images:
            full_img = '../dataset/' + img.strip()
            res = analyzer.analyze_single_image(full_img)
            print(f"Image {img}:")
            print("text_detected:", res.get("text_detected"))
