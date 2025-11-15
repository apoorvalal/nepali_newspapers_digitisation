#!/usr/bin/env python3
"""
Visualize layout regions detected by LayoutPredictor.

Shows how the page is chunked into regions for reading order detection.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings

# Test files
test_pdfs = [
    ("pdf_samples/english/TKP_2009_01_08.pdf", "English"),
    ("pdf_samples/nepali/KPUR_2009_01_05.pdf", "Nepali"),
]

output_dir = Path("ocr_output/layout_visualization")
output_dir.mkdir(parents=True, exist_ok=True)

print("Initializing LayoutPredictor...")
layout_predictor = LayoutPredictor(
    FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
)

# Color coding for different region types
COLORS = {
    'Text': 'blue',
    'SectionHeader': 'red',
    'PageHeader': 'purple',
    'Picture': 'green',
    'Figure': 'green',
    'Caption': 'orange',
    'Table': 'cyan',
}

for pdf_path, lang in test_pdfs:
    pdf_path = Path(pdf_path)
    print(f"\n{'='*70}")
    print(f"Processing: {pdf_path.name} ({lang})")
    print(f"{'='*70}")

    # Load PDF
    images, names = load_from_file(str(pdf_path))

    # Process first 3 pages
    images = images[:3]
    names = names[:3]

    print(f"Running layout detection on {len(images)} pages...")
    layout_predictions = layout_predictor(images)

    # Visualize each page
    for page_idx, (image, layout_pred, name) in enumerate(zip(images, layout_predictions, names)):
        print(f"\nPage {page_idx + 1}: {len(layout_pred.bboxes)} regions detected")

        # Convert to PIL Image for drawing
        image_pil = image.copy()
        draw = ImageDraw.Draw(image_pil)

        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Draw each layout region
        for idx, layout_box in enumerate(layout_pred.bboxes):
            bbox = layout_box.bbox  # [x1, y1, x2, y2]
            label = layout_box.label
            position = layout_box.position

            # Get color for this region type
            color = COLORS.get(label, 'yellow')

            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=5)

            # Draw position number (large, top-left of region)
            text_x = bbox[0] + 10
            text_y = bbox[1] + 10

            # Draw background rectangle for text
            text_bbox = draw.textbbox((text_x, text_y), str(position), font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((text_x, text_y), str(position), fill='white', font=font)

            # Draw label (small, top-right of region)
            label_x = bbox[2] - 150
            label_y = bbox[1] + 10
            label_bbox = draw.textbbox((label_x, label_y), label, font=small_font)
            draw.rectangle(label_bbox, fill=color)
            draw.text((label_x, label_y), label, fill='white', font=small_font)

        # Save visualization
        output_file = output_dir / f"{pdf_path.stem}_page{page_idx + 1}_layout.png"
        image_pil.save(output_file)
        print(f"  Saved: {output_file}")

        # Print region summary
        region_types = {}
        for box in layout_pred.bboxes:
            region_types[box.label] = region_types.get(box.label, 0) + 1

        print(f"  Region types: {dict(region_types)}")

print(f"\n{'='*70}")
print("DONE! Check the visualizations to see how pages are chunked.")
print(f"Output directory: {output_dir}")
print(f"{'='*70}")
print("\nColor coding:")
for label, color in COLORS.items():
    print(f"  {color:8s} = {label}")
print("\nNumbers inside boxes = reading order position")
