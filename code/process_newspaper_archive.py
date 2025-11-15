#!/usr/bin/env python3
"""
Production OCR processing script for Nepali newspaper archive.

Usage:
    python code/process_newspaper_archive.py "newspapers_archive/Kantipur"
    python code/process_newspaper_archive.py "newspapers_archive/The Kathmandu Post"

Features:
- Recursively processes all PDFs in given directory
- Optimal RTX 5070 config: 32 pages per batch, 3.4s/page
- Creates parallel directory structure in ocr_output/
- Resume capability: skips already processed files
- Progress tracking with ETA
- Error handling and logging
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import torch
from tqdm import tqdm

from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.layout import LayoutPredictor
from surya.common.surya.schema import TaskNames
from surya.settings import settings


# Optimal RTX 5070 configuration (from benchmarking)
PAGE_BATCH_SIZE = 32  # Process 32 pages at once
DETECTION_BATCH_SIZE = 12  # Internal detection batching
RECOGNITION_BATCH_SIZE = 256  # Internal recognition batching
EXPECTED_TIME_PER_PAGE = 3.4  # Seconds (measured)


def find_all_pdfs(root_dir: Path) -> List[Path]:
    """Recursively find all PDF files in directory."""
    pdf_files = []
    for path in root_dir.rglob("*.pdf"):
        if path.is_file():
            pdf_files.append(path)
    return sorted(pdf_files)


def get_output_path(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    """
    Create parallel output path structure.

    Example:
        Input:  newspapers_archive/Kantipur/2009/01/KPUR_2009_01_05.pdf
        Output: ocr_output/Kantipur/2009/01/KPUR_2009_01_05.txt
    """
    relative_path = pdf_path.relative_to(input_root)
    output_path = output_root / relative_path.with_suffix(".txt")
    return output_path


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

    # Try center point containment first
    for idx, (region_bbox, position) in enumerate(layout_regions):
        if (region_bbox[0] <= ocr_center[0] <= region_bbox[2] and
            region_bbox[1] <= ocr_center[1] <= region_bbox[3]):
            return idx, position

    # Fallback: find region with maximum overlap
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


def sort_text_by_reading_order(ocr_predictions, layout_predictions):
    """
    Sort OCR text lines according to layout reading order.

    This handles multi-column newspaper layouts by using the LayoutPredictor
    to determine proper reading sequence across columns.
    """
    sorted_pages = []

    for page_idx, (ocr_pred, layout_pred) in enumerate(zip(ocr_predictions, layout_predictions)):
        # Prepare layout regions (bbox coordinates and reading position)
        layout_regions = [(box.bbox, box.position) for box in layout_pred.bboxes]

        # Assign each OCR line to a layout region
        text_lines_with_order = []
        for line in ocr_pred.text_lines:
            region_idx, position = find_containing_region(line.bbox, layout_regions)
            text_lines_with_order.append({
                'text': line.text,
                'bbox': line.bbox,
                'reading_position': position,
                'y_coord': line.bbox[1]  # for sorting within region
            })

        # Sort by reading position, then by y-coordinate within same region
        text_lines_with_order.sort(key=lambda x: (x['reading_position'], x['y_coord']))

        sorted_pages.append(text_lines_with_order)

    return sorted_pages


def extract_text_from_sorted_predictions(sorted_pages) -> str:
    """Extract plain text from sorted OCR predictions."""
    text_lines = []

    for page_idx, page_lines in enumerate(sorted_pages):
        text_lines.append(f"\n{'=' * 60}")
        text_lines.append(f"PAGE {page_idx + 1}")
        text_lines.append(f"{'=' * 60}\n")

        for item in page_lines:
            text_lines.append(item['text'])

    return "\n".join(text_lines)


def save_text_output(text: str, output_path: Path, metadata: Dict):
    """Save OCR text with metadata header."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Write metadata header
        f.write(f"{'=' * 60}\n")
        f.write(f"OCR Metadata\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Source PDF: {metadata['source_pdf']}\n")
        f.write(f"Pages: {metadata['num_pages']}\n")
        f.write(f"Processing time: {metadata['processing_time']:.1f}s\n")
        f.write(f"Time per page: {metadata['time_per_page']:.1f}s\n")
        f.write(f"Processed: {metadata['timestamp']}\n")
        f.write(f"GPU: {metadata['gpu']}\n")
        f.write(
            f"Config: batch_size={metadata['batch_size']}, det_batch={metadata['det_batch']}, rec_batch={metadata['rec_batch']}\n"
        )
        f.write(f"{'=' * 60}\n\n")

        # Write OCR text
        f.write(text)


def process_pdf(
    pdf_path: Path,
    output_path: Path,
    foundation_predictor,
    det_predictor,
    rec_predictor,
    layout_predictor,
) -> Dict:
    """
    Process a single PDF with OCR and reading order detection.

    Returns metadata dict with processing stats.
    """
    start_time = time.time()

    # Load PDF
    images, names = load_from_file(str(pdf_path))
    num_pages = len(images)

    # Step 1: Detect layout and reading order
    # (This is NOT batched because layout models are fast and full-page context is important)
    layout_predictions = layout_predictor(images)

    # Step 2: Process all pages with OCR in optimal batches
    all_predictions = []

    for i in range(0, num_pages, PAGE_BATCH_SIZE):
        batch_images = images[i : i + PAGE_BATCH_SIZE]

        predictions = rec_predictor(
            batch_images,
            task_names=[TaskNames.ocr_with_boxes] * len(batch_images),
            det_predictor=det_predictor,
            math_mode=False,
            detection_batch_size=DETECTION_BATCH_SIZE,
            recognition_batch_size=RECOGNITION_BATCH_SIZE,
        )

        all_predictions.extend(predictions)

    # Step 3: Sort text by reading order (handles multi-column layouts)
    sorted_pages = sort_text_by_reading_order(all_predictions, layout_predictions)

    # Extract text
    text = extract_text_from_sorted_predictions(sorted_pages)

    # Calculate stats
    elapsed = time.time() - start_time
    time_per_page = elapsed / num_pages

    # Create metadata
    metadata = {
        "source_pdf": str(pdf_path),
        "num_pages": num_pages,
        "processing_time": elapsed,
        "time_per_page": time_per_page,
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "batch_size": PAGE_BATCH_SIZE,
        "det_batch": DETECTION_BATCH_SIZE,
        "rec_batch": RECOGNITION_BATCH_SIZE,
    }

    # Save output
    save_text_output(text, output_path, metadata)

    return metadata


def process_directory(input_dir: str, output_base: str = "ocr_output"):
    """
    Process all PDFs in a directory with progress tracking and resume capability.
    """
    input_root = Path(input_dir).resolve()
    output_root = Path(output_base).resolve()

    if not input_root.exists():
        print(f"Error: Input directory does not exist: {input_root}")
        sys.exit(1)

    # Find all PDFs
    print(f"\nScanning for PDFs in: {input_root}")
    pdf_files = find_all_pdfs(input_root)

    if not pdf_files:
        print(f"No PDF files found in {input_root}")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF files")

    # Filter out already processed files
    to_process = []
    already_done = []

    for pdf_path in pdf_files:
        output_path = get_output_path(pdf_path, input_root, output_root)
        if output_path.exists():
            already_done.append(pdf_path)
        else:
            to_process.append(pdf_path)

    if already_done:
        print(f"Skipping {len(already_done)} already processed files")

    if not to_process:
        print("All files already processed!")
        sys.exit(0)

    print(f"\nProcessing {len(to_process)} PDFs")
    print(f"Output directory: {output_root}")
    print(f"\nOptimal RTX 5070 Config:")
    print(f"  Batch size: {PAGE_BATCH_SIZE} pages")
    print(f"  Detection batch: {DETECTION_BATCH_SIZE}")
    print(f"  Recognition batch: {RECOGNITION_BATCH_SIZE}")
    print(f"  Expected: {EXPECTED_TIME_PER_PAGE:.1f}s per page")

    # Estimate total time
    total_pages_estimate = len(to_process) * 12  # Assume avg 12 pages per PDF
    estimated_seconds = total_pages_estimate * EXPECTED_TIME_PER_PAGE
    estimated_time = str(timedelta(seconds=int(estimated_seconds)))

    print(f"\nEstimated total time: {estimated_time}")
    print(f"{'=' * 60}\n")

    # Initialize OCR models
    print("Loading OCR models...")
    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)
    layout_predictor = LayoutPredictor(
        FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    )
    print("Models loaded!\n")

    # Process PDFs
    total_pages = 0
    total_time = 0
    errors = []

    start_time = time.time()

    for pdf_path in tqdm(to_process, desc="Processing PDFs", unit="pdf"):
        try:
            output_path = get_output_path(pdf_path, input_root, output_root)

            metadata = process_pdf(
                pdf_path,
                output_path,
                foundation_predictor,
                det_predictor,
                rec_predictor,
                layout_predictor,
            )

            total_pages += metadata["num_pages"]
            total_time += metadata["processing_time"]

            # Update progress bar with stats
            avg_time_per_page = total_time / total_pages if total_pages > 0 else 0
            tqdm.write(
                f"  {pdf_path.name}: {metadata['num_pages']} pages, "
                f"{metadata['time_per_page']:.1f}s/page"
            )

        except Exception as e:
            error_msg = f"{pdf_path}: {str(e)}"
            errors.append(error_msg)
            tqdm.write(f"  ERROR: {error_msg}")
            continue

    # Final summary
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"PDFs processed: {len(to_process) - len(errors)}/{len(to_process)}")
    print(f"Total pages: {total_pages}")
    print(f"Total time: {str(timedelta(seconds=int(elapsed)))}")
    print(f"Average: {total_time / total_pages:.1f}s per page")

    if errors:
        print(f"\nErrors: {len(errors)}")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print(f"\nOutput saved to: {output_root}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python code/process_newspaper_archive.py <input_directory>")
        print("\nExamples:")
        print(
            '  python code/process_newspaper_archive.py "newspapers_archive/Kantipur"'
        )
        print(
            '  python code/process_newspaper_archive.py "newspapers_archive/The Kathmandu Post"'
        )
        sys.exit(1)

    input_directory = sys.argv[1]

    # Optional: custom output directory
    output_directory = sys.argv[2] if len(sys.argv) > 2 else "ocr_output"

    # Set environment variable for optimal memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    process_directory(input_directory, output_directory)
