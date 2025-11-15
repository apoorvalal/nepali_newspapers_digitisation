#!/usr/bin/env python3
"""
Test reading order detection with Nepali (Devanagari) newspaper.
"""

from pathlib import Path
from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames
from surya.settings import settings

# Test with Kantipur (Nepali newspaper)
pdf_path = Path("pdf_samples/nepali/KPUR_2009_01_05.pdf")
output_dir = Path("ocr_output/reading_order_test")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading PDF: {pdf_path}")
images, names = load_from_file(str(pdf_path))

# Only process first page
images = images[:1]
names = names[:1]

print("Initializing predictors...")
foundation_predictor = FoundationPredictor()
layout_predictor = LayoutPredictor(
    FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
)
det_predictor = DetectionPredictor()
rec_predictor = RecognitionPredictor(foundation_predictor)

print("Step 1: Detecting layout and reading order...")
layout_predictions = layout_predictor(images)

print("Step 2: Running OCR...")
ocr_predictions = rec_predictor(
    images,
    task_names=[TaskNames.ocr_with_boxes] * len(images),
    det_predictor=det_predictor,
)

layout_pred = layout_predictions[0]
ocr_pred = ocr_predictions[0]

print(f"\nLayout analysis found {len(layout_pred.bboxes)} regions")
print("Region types:", set([bbox.label for bbox in layout_pred.bboxes]))

def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def bbox_overlap_area(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0

def find_containing_region(ocr_bbox, layout_regions):
    ocr_center = bbox_center(ocr_bbox)

    # Try center point containment
    for idx, (region_bbox, position) in enumerate(layout_regions):
        if (region_bbox[0] <= ocr_center[0] <= region_bbox[2] and
            region_bbox[1] <= ocr_center[1] <= region_bbox[3]):
            return idx, position

    # Fallback: maximum overlap
    max_overlap = 0
    best_region = None
    best_position = 999

    for idx, (region_bbox, position) in enumerate(layout_regions):
        overlap = bbox_overlap_area(ocr_bbox, region_bbox)
        if overlap > max_overlap:
            max_overlap = overlap
            best_region = idx
            best_position = position

    if best_region is not None:
        return best_region, best_position

    return -1, 999

layout_regions = [(box.bbox, box.position) for box in layout_pred.bboxes]

text_lines_with_order = []
for line in ocr_pred.text_lines:
    region_idx, position = find_containing_region(line.bbox, layout_regions)
    text_lines_with_order.append({
        'text': line.text,
        'bbox': line.bbox,
        'confidence': line.confidence,
        'region_idx': region_idx,
        'reading_position': position,
        'y_coord': line.bbox[1]
    })

text_lines_with_order.sort(key=lambda x: (x['reading_position'], x['y_coord']))

# Write sorted text
sorted_text_file = output_dir / f"{names[0]}_sorted_text.txt"
with open(sorted_text_file, 'w', encoding='utf-8') as f:
    f.write(f"Text in Reading Order - {names[0]}\n")
    f.write("=" * 70 + "\n\n")

    current_position = None
    for item in text_lines_with_order:
        if item['reading_position'] != current_position:
            if current_position is not None:
                f.write("\n" + "-" * 70 + "\n\n")
            current_position = item['reading_position']
            f.write(f"[Region Position {current_position}]\n\n")

        f.write(item['text'] + "\n")

print(f"Wrote sorted text to: {sorted_text_file}")
print("\nDONE! Check the output to verify Devanagari text is properly ordered.")
