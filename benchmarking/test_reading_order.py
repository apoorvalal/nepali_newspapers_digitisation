#!/usr/bin/env python3
"""
Test reading order detection for multi-column newspaper layouts.

This script demonstrates how to use Surya's LayoutPredictor to detect
reading order in multi-column documents, then combine with OCR results
to extract text in the correct sequence.
"""

from pathlib import Path
from PIL import Image
from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames
from surya.settings import settings

# Test with the English newspaper sample
pdf_path = Path("pdf_samples/english/TKP_2009_01_08.pdf")
output_dir = Path("ocr_output/reading_order_test")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading PDF: {pdf_path}")
images, names = load_from_file(str(pdf_path))

# Only process first page for testing
images = images[:1]
names = names[:1]

print("Initializing predictors...")
foundation_predictor = FoundationPredictor()

# Layout predictor for reading order
layout_predictor = LayoutPredictor(
    FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
)

# OCR predictors
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

# Process results for first page
layout_pred = layout_predictions[0]
ocr_pred = ocr_predictions[0]

print(f"\nLayout analysis found {len(layout_pred.bboxes)} regions")
print("Region types:", set([bbox.label for bbox in layout_pred.bboxes]))

# Write layout information
layout_file = output_dir / f"{names[0]}_layout.txt"
with open(layout_file, 'w', encoding='utf-8') as f:
    f.write(f"Layout Analysis - {names[0]}\n")
    f.write("=" * 70 + "\n\n")

    for idx, layout_box in enumerate(layout_pred.bboxes):
        f.write(f"[{idx}] Position in reading order: {layout_box.position}\n")
        f.write(f"    Label: {layout_box.label}\n")
        f.write(f"    BBox: {layout_box.bbox}\n\n")

print(f"Wrote layout info to: {layout_file}")

# Write OCR results sorted by layout position
# Strategy: For each OCR line, find which layout region it belongs to
# Then sort by region position first, then by y-coordinate within region

def bbox_center(bbox):
    """Calculate center point of bounding box."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def bbox_overlap_area(bbox1, bbox2):
    """Calculate overlap area between two bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0

def find_containing_region(ocr_bbox, layout_regions):
    """Find which layout region contains this OCR bounding box."""
    ocr_center = bbox_center(ocr_bbox)

    # First, try to find region by center point containment
    for idx, (region_bbox, position) in enumerate(layout_regions):
        if (region_bbox[0] <= ocr_center[0] <= region_bbox[2] and
            region_bbox[1] <= ocr_center[1] <= region_bbox[3]):
            return idx, position

    # Fallback: find region with maximum overlap
    max_overlap = 0
    best_region = None
    best_position = 999  # default high number

    for idx, (region_bbox, position) in enumerate(layout_regions):
        overlap = bbox_overlap_area(ocr_bbox, region_bbox)
        if overlap > max_overlap:
            max_overlap = overlap
            best_region = idx
            best_position = position

    if best_region is not None:
        return best_region, best_position

    # Last fallback: return high position number (will sort last)
    return -1, 999

# Prepare layout regions (bbox coordinates and position)
layout_regions = [(box.bbox, box.position) for box in layout_pred.bboxes]

# Assign each OCR line to a layout region and position
text_lines_with_order = []
for line in ocr_pred.text_lines:
    region_idx, position = find_containing_region(line.bbox, layout_regions)
    text_lines_with_order.append({
        'text': line.text,
        'bbox': line.bbox,
        'confidence': line.confidence,
        'region_idx': region_idx,
        'reading_position': position,
        'y_coord': line.bbox[1]  # for sorting within region
    })

# Sort by reading position, then by y-coordinate within same region
text_lines_with_order.sort(key=lambda x: (x['reading_position'], x['y_coord']))

# Write sorted text output
sorted_text_file = output_dir / f"{names[0]}_sorted_text.txt"
with open(sorted_text_file, 'w', encoding='utf-8') as f:
    f.write(f"Text in Reading Order - {names[0]}\n")
    f.write("=" * 70 + "\n\n")

    current_position = None
    for idx, item in enumerate(text_lines_with_order):
        # Add separator when moving to new region
        if item['reading_position'] != current_position:
            if current_position is not None:
                f.write("\n" + "-" * 70 + "\n\n")
            current_position = item['reading_position']
            f.write(f"[Region Position {current_position}]\n\n")

        f.write(item['text'] + "\n")

print(f"Wrote sorted text to: {sorted_text_file}")

# Also write unsorted (original sequential) for comparison
unsorted_text_file = output_dir / f"{names[0]}_unsorted_text.txt"
with open(unsorted_text_file, 'w', encoding='utf-8') as f:
    f.write(f"Text in Sequential Order (UNSORTED) - {names[0]}\n")
    f.write("=" * 70 + "\n\n")

    for line in ocr_pred.text_lines:
        f.write(line.text + "\n")

print(f"Wrote unsorted text to: {unsorted_text_file}")

print("\n" + "=" * 70)
print("DONE! Compare the sorted vs unsorted text files to see the difference.")
print("Sorted text should have coherent articles, while unsorted will be jumbled.")
print("=" * 70)
