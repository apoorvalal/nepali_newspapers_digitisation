#!/usr/bin/env python
"""
Quick test of Surya OCR on a single page from each sample PDF.
"""

import os
import json
import time
from pathlib import Path

from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames


def test_single_page(pdf_path: str, page_index: int = 0):
    """Test OCR on a single page from a PDF."""
    print(f"\n{'='*60}")
    print(f"Testing: {pdf_path} (page {page_index + 1})")
    print(f"{'='*60}\n")

    # Load PDF
    print("[1/3] Loading PDF...")
    images, names = load_from_file(pdf_path)
    print(f"  Loaded {len(images)} total pages")

    # Use only the specified page
    test_image = [images[page_index]]
    test_name = names[page_index]
    print(f"  Testing page: {test_name}\n")

    # Initialize models
    print("[2/3] Initializing models...")
    start = time.time()
    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)
    print(f"  Loaded in {time.time() - start:.2f}s\n")

    # Run OCR
    print("[3/3] Running OCR...")
    start_ocr = time.time()
    predictions = rec_predictor(
        test_image,
        task_names=[TaskNames.ocr_with_boxes],
        det_predictor=det_predictor,
        math_mode=False,
        detection_batch_size=6,
        recognition_batch_size=128
    )
    ocr_time = time.time() - start_ocr
    print(f"  Completed in {ocr_time:.2f}s\n")

    # Show results
    pred = predictions[0]
    print("="*60)
    print(f"Results for {Path(pdf_path).name} - Page {page_index + 1}")
    print("="*60)
    print(f"Text lines detected: {len(pred.text_lines)}")
    print(f"Total characters: {sum(len(line.text) for line in pred.text_lines)}")
    print(f"\nFirst 10 lines:\n")

    for i, line in enumerate(pred.text_lines[:10]):
        preview = line.text[:100] + "..." if len(line.text) > 100 else line.text
        print(f"{i+1:2d}. {preview}")

    print(f"\n{'='*60}\n")

    # Save to JSON
    output_dir = "ocr_output/quick_test"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{Path(pdf_path).stem}_page{page_index+1}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pred.model_dump(), f, ensure_ascii=False, indent=2)

    print(f"Full results saved to: {output_file}\n")


def main():
    # Test first page of English newspaper
    print("\n" + "="*60)
    print(" ENGLISH NEWSPAPER TEST")
    print("="*60)
    test_single_page("pdf_samples/english/TKP_2009_01_08.pdf", page_index=0)

    # Test first page of Nepali newspaper
    print("\n" + "="*60)
    print(" NEPALI NEWSPAPER TEST")
    print("="*60)
    test_single_page("pdf_samples/nepali/KPUR_2009_01_05.pdf", page_index=0)


if __name__ == "__main__":
    main()
