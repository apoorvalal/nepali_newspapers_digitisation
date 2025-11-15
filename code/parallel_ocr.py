#!/usr/bin/env python
"""
Parallel OCR processing using pdftk to split PDFs and multiprocessing.
"""

import os
import json
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple

from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames


def split_pdf_with_pdftk(pdf_path: str, output_dir: str) -> List[str]:
    """
    Split a PDF into individual pages using pdftk.
    Returns list of single-page PDF paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get page count
    result = subprocess.run(
        ["pdftk", pdf_path, "dump_data"],
        capture_output=True,
        text=True
    )

    num_pages = 0
    for line in result.stdout.split("\n"):
        if line.startswith("NumberOfPages:"):
            num_pages = int(line.split(":")[1].strip())
            break

    if num_pages == 0:
        raise ValueError(f"Could not determine page count for {pdf_path}")

    # Split into individual pages
    page_files = []
    base_name = Path(pdf_path).stem

    for page_num in range(1, num_pages + 1):
        output_file = os.path.join(output_dir, f"{base_name}_page_{page_num:04d}.pdf")

        subprocess.run(
            ["pdftk", pdf_path, "cat", str(page_num), "output", output_file],
            check=True,
            capture_output=True
        )

        page_files.append(output_file)

    return page_files


def process_single_page_pdf(args: Tuple[str, int]) -> Dict:
    """
    Process a single-page PDF file.
    Returns OCR results as dict.
    """
    pdf_path, page_num = args

    try:
        # Load the single page
        images, names = load_from_file(pdf_path)

        if len(images) != 1:
            raise ValueError(f"Expected 1 page, got {len(images)}")

        # Initialize predictors (cached per process)
        if not hasattr(process_single_page_pdf, '_predictors'):
            foundation_predictor = FoundationPredictor()
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor(foundation_predictor)
            process_single_page_pdf._predictors = (det_predictor, rec_predictor)

        det_predictor, rec_predictor = process_single_page_pdf._predictors

        # Run OCR
        start_time = time.time()
        predictions = rec_predictor(
            images,
            task_names=[TaskNames.ocr_with_boxes],
            det_predictor=det_predictor,
            math_mode=False,
            detection_batch_size=6,
            recognition_batch_size=256
        )

        result = predictions[0].model_dump()
        result['page_number'] = page_num
        result['processing_time'] = time.time() - start_time
        result['source_pdf'] = pdf_path

        return {
            'success': True,
            'page_number': page_num,
            'result': result,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'page_number': page_num,
            'result': None,
            'error': str(e)
        }


def process_pdf_parallel(pdf_path: str, num_workers: int = 2, output_dir: str = "ocr_output/parallel"):
    """
    Process a PDF using parallel workers.

    Args:
        pdf_path: Path to PDF file
        num_workers: Number of parallel workers (default 2 for dual GPU or CPU cores)
        output_dir: Directory to save results
    """
    print(f"\n{'='*70}")
    print(f"PARALLEL PROCESSING: {Path(pdf_path).name}")
    print(f"Workers: {num_workers}")
    print(f"{'='*70}\n")

    # Create temp directory for split PDFs
    with tempfile.TemporaryDirectory() as temp_dir:
        print("[1/4] Splitting PDF with pdftk...")
        start_split = time.time()
        page_files = split_pdf_with_pdftk(pdf_path, temp_dir)
        split_time = time.time() - start_split
        print(f"  Split into {len(page_files)} pages in {split_time:.2f}s\n")

        # Prepare arguments for parallel processing
        args_list = [(page_file, i+1) for i, page_file in enumerate(page_files)]

        print(f"[2/4] Processing {len(page_files)} pages with {num_workers} workers...")
        start_ocr = time.time()

        # Process in parallel
        with Pool(processes=num_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap(process_single_page_pdf, args_list), 1):
                results.append(result)
                if result['success']:
                    lines = len(result['result']['text_lines'])
                    proc_time = result['result']['processing_time']
                    print(f"  Page {i}/{len(page_files)} done: {lines} lines in {proc_time:.1f}s")
                else:
                    print(f"  Page {i}/{len(page_files)} FAILED: {result['error']}")

        ocr_time = time.time() - start_ocr
        print(f"\n  OCR completed in {ocr_time:.1f}s ({len(page_files)/ocr_time:.2f} pages/sec)\n")

        print("[3/4] Aggregating results...")
        successful_results = [r['result'] for r in results if r['success']]
        failed_pages = [r['page_number'] for r in results if not r['success']]

        total_lines = sum(len(r['text_lines']) for r in successful_results)
        total_chars = sum(sum(len(line['text']) for line in r['text_lines']) for r in successful_results)

        print(f"  Success: {len(successful_results)}/{len(page_files)} pages")
        if failed_pages:
            print(f"  Failed pages: {failed_pages}")
        print(f"  Total lines: {total_lines}")
        print(f"  Total characters: {total_chars}\n")

        print("[4/4] Saving results...")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{Path(pdf_path).stem}_parallel.json")

        output_data = {
            'source_pdf': pdf_path,
            'total_pages': len(page_files),
            'successful_pages': len(successful_results),
            'failed_pages': failed_pages,
            'total_lines': total_lines,
            'total_characters': total_chars,
            'split_time_seconds': split_time,
            'ocr_time_seconds': ocr_time,
            'total_time_seconds': split_time + ocr_time,
            'pages_per_second': len(page_files) / ocr_time,
            'pages': successful_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"  Saved to: {output_file}\n")

        print(f"{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Total time: {split_time + ocr_time:.1f}s")
        print(f"  - Splitting: {split_time:.1f}s")
        print(f"  - OCR: {ocr_time:.1f}s")
        print(f"Throughput: {len(page_files)/ocr_time:.2f} pages/sec")
        print(f"{'='*70}\n")

        return output_data


def main():
    # Test with English PDF
    pdf_path = "pdf_samples/english/TKP_2009_01_08.pdf"

    # Try with 2 workers first (conservative)
    print("Testing with 2 parallel workers...")
    result = process_pdf_parallel(pdf_path, num_workers=2)

    print(f"\nEstimated time for full archive with 2 workers:")
    total_pages = 26427 * 12
    hours = total_pages / result['pages_per_second'] / 3600
    days = hours / 24
    print(f"  ~{hours:.0f} hours = {days:.1f} days\n")


if __name__ == "__main__":
    main()
