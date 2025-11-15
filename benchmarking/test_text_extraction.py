#!/usr/bin/env python3
"""
Test text extraction quality on sample PDFs.

Processes a few sample PDFs and saves extracted text to review quality
before committing to full production run.
"""

import os
from pathlib import Path
from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 3) -> str:
    """Extract text from first N pages of PDF."""

    # Load PDF (limit to first N pages for quick test)
    images, names = load_from_file(str(pdf_path))
    images = images[:max_pages]

    # Initialize models
    print(f"Processing {pdf_path.name} ({len(images)} pages)...")
    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    # Run OCR
    predictions = rec_predictor(
        images,
        task_names=[TaskNames.ocr_with_boxes] * len(images),
        det_predictor=det_predictor,
        math_mode=False,
        detection_batch_size=12,
        recognition_batch_size=256
    )

    # Extract text
    text_lines = []
    for page_idx, pred in enumerate(predictions):
        text_lines.append(f"\n{'='*70}")
        text_lines.append(f"PAGE {page_idx + 1}")
        text_lines.append(f"{'='*70}\n")

        for line in pred.text_lines:
            text_lines.append(line.text)

    return '\n'.join(text_lines)


if __name__ == "__main__":
    # Set environment
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Output directory
    output_dir = Path("ocr_output/text_quality_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test PDFs
    test_pdfs = [
        Path("pdf_samples/english/TKP_2009_01_08.pdf"),   # English
        Path("pdf_samples/nepali/KPUR_2009_01_05.pdf"),   # Nepali
    ]

    print("="*70)
    print("TEXT EXTRACTION QUALITY TEST")
    print("="*70)
    print(f"\nExtracting text from {len(test_pdfs)} sample PDFs")
    print(f"Output: {output_dir}\n")

    for pdf_path in test_pdfs:
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found, skipping...")
            continue

        # Extract text
        text = extract_text_from_pdf(pdf_path, max_pages=3)

        # Save to file
        output_file = output_dir / f"{pdf_path.stem}_text.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Source: {pdf_path}\n")
            f.write(f"{'='*70}\n\n")
            f.write(text)

        print(f"  Saved: {output_file}")

        # Show preview
        lines = text.split('\n')
        content_lines = [l for l in lines if l.strip() and not l.startswith('=')]
        preview = content_lines[:20]

        print(f"\n  Preview (first 20 lines):")
        for line in preview:
            print(f"    {line}")
        print()

    print(f"\n{'='*70}")
    print("COMPLETE - Review files in ocr_output/text_quality_test/")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the extracted text quality")
    print("2. Check both English and Nepali accuracy")
    print("3. If satisfied, run production script on a single publication")
