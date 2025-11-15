#!/usr/bin/env python
"""
Test script for Surya OCR on sample newspaper PDFs.
Based on the Surya OCR library examples.
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


def test_ocr_on_pdf(pdf_path: str, output_dir: str = "ocr_output", batch_size: int = 1):
    """
    Run Surya OCR on a single PDF file and save results.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output results
        batch_size: Number of pages to process at once (default 1 for 8GB GPU)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"{'='*60}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load PDF into images
    print("\n[1/4] Loading PDF...")
    start_load = time.time()
    images, names = load_from_file(pdf_path)
    print(f"  Loaded {len(images)} page(s) in {time.time() - start_load:.2f}s")

    # Initialize predictors (models will download on first run)
    print("\n[2/4] Initializing Surya models...")
    start_init = time.time()
    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)
    print(f"  Models loaded in {time.time() - start_init:.2f}s")

    # Run OCR in batches to avoid GPU OOM
    print(f"\n[3/4] Running OCR (batch size: {batch_size})...")
    start_ocr = time.time()
    predictions = []

    for i in range(0, len(images), batch_size):
        batch_end = min(i + batch_size, len(images))
        batch_images = images[i:batch_end]
        batch_task_names = [TaskNames.ocr_with_boxes] * len(batch_images)

        print(f"  Processing pages {i+1}-{batch_end}/{len(images)}...")
        batch_preds = rec_predictor(
            batch_images,
            task_names=batch_task_names,
            det_predictor=det_predictor,
            math_mode=False,  # Disable math recognition for newspapers
            detection_batch_size=6,  # Reduce batch size for detection
            recognition_batch_size=128  # Reduce batch size for recognition
        )
        predictions.extend(batch_preds)

    ocr_time = time.time() - start_ocr
    print(f"  OCR completed in {ocr_time:.2f}s")

    # Process results
    print("\n[4/4] Processing results...")
    total_lines = 0
    total_chars = 0

    for idx, (name, pred) in enumerate(zip(names, predictions)):
        page_lines = len(pred.text_lines)
        page_chars = sum(len(line.text) for line in pred.text_lines)
        total_lines += page_lines
        total_chars += page_chars

        print(f"\n  Page {idx + 1} ({name}):")
        print(f"    Lines detected: {page_lines}")
        print(f"    Characters: {page_chars}")

        # Print first few lines as sample
        print(f"    Sample text (first 3 lines):")
        for i, line in enumerate(pred.text_lines[:3]):
            preview = line.text[:80] + "..." if len(line.text) > 80 else line.text
            print(f"      {i+1}. {preview}")

    # Save results to JSON
    output_file = os.path.join(output_dir, f"{Path(pdf_path).stem}_ocr.json")
    results = []
    for name, pred in zip(names, predictions):
        page_result = pred.model_dump()
        page_result["page_name"] = name
        results.append(page_result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total pages: {len(images)}")
    print(f"  Total lines: {total_lines}")
    print(f"  Total characters: {total_chars}")
    print(f"  Processing time: {ocr_time:.2f}s")
    print(f"  Speed: {total_chars / ocr_time:.1f} chars/sec")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*60}\n")

    return results


def main():
    """Test OCR on sample PDFs."""

    # Test on English PDF
    english_pdf = "pdf_samples/english/TKP_2009_01_08.pdf"
    if os.path.exists(english_pdf):
        print("\nTesting on ENGLISH newspaper...")
        test_ocr_on_pdf(english_pdf, "ocr_output/english")

    # Test on Nepali PDF
    nepali_pdf = "pdf_samples/nepali/KPUR_2009_01_05.pdf"
    if os.path.exists(nepali_pdf):
        print("\nTesting on NEPALI newspaper...")
        test_ocr_on_pdf(nepali_pdf, "ocr_output/nepali")


if __name__ == "__main__":
    main()
