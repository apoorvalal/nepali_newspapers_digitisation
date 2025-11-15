#!/usr/bin/env python3
"""
Visualize OCR bounding boxes overlaid on PDF pages.

Helps debug detection quality and identify hyperparameter issues.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames


def visualize_bboxes(pdf_path: Path, output_dir: Path, max_pages: int = 3):
    """
    Run OCR and visualize bounding boxes on PDF pages.
    """

    # Load PDF
    print(f"\nProcessing: {pdf_path.name}")
    images, names = load_from_file(str(pdf_path))
    images = images[:max_pages]

    # Initialize models
    print("Loading models...")
    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    # Run OCR
    print(f"Running OCR on {len(images)} pages...")
    predictions = rec_predictor(
        images,
        task_names=[TaskNames.ocr_with_boxes] * len(images),
        det_predictor=det_predictor,
        math_mode=False,
        detection_batch_size=12,
        recognition_batch_size=256
    )

    # Visualize each page
    output_dir.mkdir(parents=True, exist_ok=True)

    for page_idx, (image, pred) in enumerate(zip(images, predictions)):
        print(f"\nPage {page_idx + 1}:")
        print(f"  Detected {len(pred.text_lines)} text lines")

        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image.copy()

        # Create drawing context
        draw = ImageDraw.Draw(image_pil)

        # Draw each bounding box
        for idx, line in enumerate(pred.text_lines):
            # Get bbox coordinates
            bbox = line.bbox  # [x1, y1, x2, y2]

            # Draw rectangle
            draw.rectangle(bbox, outline='red', width=2)

            # Draw line number
            draw.text((bbox[0], bbox[1] - 15), str(idx), fill='blue')

        # Save annotated image
        output_file = output_dir / f"{pdf_path.stem}_page{page_idx + 1}_bboxes.png"
        image_pil.save(output_file)
        print(f"  Saved: {output_file}")

        # Also save text with line numbers for reference
        text_file = output_dir / f"{pdf_path.stem}_page{page_idx + 1}_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Page {page_idx + 1} - Text Lines with Bounding Boxes\n")
            f.write("="*70 + "\n\n")

            for idx, line in enumerate(pred.text_lines):
                bbox = line.bbox
                f.write(f"[{idx}] bbox: {bbox}\n")
                f.write(f"    text: {line.text}\n")
                f.write(f"    confidence: {line.confidence:.3f}\n\n")

        print(f"  Text saved: {text_file}")

        # Print first few lines as preview
        print(f"\n  First 10 lines:")
        for idx, line in enumerate(pred.text_lines[:10]):
            print(f"    [{idx}] {line.text[:60]}...")


if __name__ == "__main__":
    # Set environment
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Output directory
    output_dir = Path("ocr_output/bbox_visualization")

    # Test PDFs
    test_pdfs = [
        Path("pdf_samples/english/TKP_2009_01_08.pdf"),
        Path("pdf_samples/nepali/KPUR_2009_01_05.pdf"),
    ]

    print("="*70)
    print("BOUNDING BOX VISUALIZATION")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Will process {len(test_pdfs)} PDFs (first 3 pages each)\n")

    for pdf_path in test_pdfs:
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found, skipping...")
            continue

        visualize_bboxes(pdf_path, output_dir, max_pages=3)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"\nCheck {output_dir}/ for:")
    print("  - *_bboxes.png - Annotated images with bounding boxes")
    print("  - *_text.txt - Extracted text with bbox coordinates")
    print("\nUse these to identify issues with text line detection.")
