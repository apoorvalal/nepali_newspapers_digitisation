#!/usr/bin/env python
"""
Test different batch sizes to find optimal throughput.
"""

import os
import time
import torch
from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames


def test_batch_processing(pdf_path: str, page_batch_size: int, detection_batch_size: int, recognition_batch_size: int):
    """Test OCR with specific batch sizes."""
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(pdf_path)}")
    print(f"Page batch: {page_batch_size} | Detection batch: {detection_batch_size} | Recognition batch: {recognition_batch_size}")
    print(f"{'='*70}\n")

    # Load PDF
    images, names = load_from_file(pdf_path)
    num_pages = min(len(images), 4)  # Test with first 4 pages
    images = images[:num_pages]
    print(f"Testing with {num_pages} pages")

    # Initialize models
    print("Initializing models...")
    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Get initial memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial GPU memory: {initial_mem:.2f} GB\n")

    try:
        start_time = time.time()
        all_predictions = []

        for i in range(0, num_pages, page_batch_size):
            batch_end = min(i + page_batch_size, num_pages)
            batch_images = images[i:batch_end]
            batch_size_actual = len(batch_images)

            print(f"Processing pages {i+1}-{batch_end}...", end=" ", flush=True)

            batch_start = time.time()
            predictions = rec_predictor(
                batch_images,
                task_names=[TaskNames.ocr_with_boxes] * batch_size_actual,
                det_predictor=det_predictor,
                math_mode=False,
                detection_batch_size=detection_batch_size,
                recognition_batch_size=recognition_batch_size
            )
            batch_time = time.time() - batch_start

            all_predictions.extend(predictions)

            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                current_mem = torch.cuda.memory_allocated() / 1024**3
                print(f"Done in {batch_time:.1f}s (Current: {current_mem:.2f}GB, Peak: {peak_mem:.2f}GB)")
            else:
                print(f"Done in {batch_time:.1f}s")

            # Clear cache between batches
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        pages_per_sec = num_pages / total_time

        print(f"\n{'='*70}")
        print(f"SUCCESS!")
        print(f"Total time: {total_time:.1f}s ({pages_per_sec:.2f} pages/sec)")
        print(f"Average per page: {total_time/num_pages:.1f}s")

        total_lines = sum(len(p.text_lines) for p in all_predictions)
        total_chars = sum(sum(len(line.text) for line in p.text_lines) for p in all_predictions)
        print(f"Total lines: {total_lines}, Total chars: {total_chars}")
        print(f"{'='*70}\n")

        return True, pages_per_sec

    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM ERROR: {str(e)[:100]}...")
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory before OOM: {peak_mem:.2f} GB")
        torch.cuda.empty_cache()
        return False, 0.0
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)[:100]}...")
        return False, 0.0


def main():
    pdf_path = "pdf_samples/english/TKP_2009_01_08.pdf"

    # Test configurations from conservative to aggressive
    configs = [
        # (page_batch, detection_batch, recognition_batch)
        (1, 6, 128),    # Current conservative settings
        (2, 6, 128),    # Try 2 pages at once
        (3, 6, 128),    # Try 3 pages
        (4, 6, 128),    # Try 4 pages
        (2, 8, 256),    # Increase detection batch
        (3, 8, 256),    # Combine both
        (4, 12, 256),   # More aggressive
    ]

    results = []

    for page_batch, det_batch, rec_batch in configs:
        success, speed = test_batch_processing(
            pdf_path,
            page_batch,
            det_batch,
            rec_batch
        )
        results.append((page_batch, det_batch, rec_batch, success, speed))

        # If we hit OOM, don't try more aggressive configs
        if not success:
            print("Hit OOM, stopping tests.\n")
            break

        # Small pause between tests
        time.sleep(2)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF BATCH SIZE TESTS")
    print("="*70)
    print(f"{'Page Batch':<12} {'Det Batch':<12} {'Rec Batch':<12} {'Status':<10} {'Speed (pg/s)':<15}")
    print("-"*70)

    for page_b, det_b, rec_b, success, speed in results:
        status = "SUCCESS" if success else "OOM"
        speed_str = f"{speed:.3f}" if success else "N/A"
        print(f"{page_b:<12} {det_b:<12} {rec_b:<12} {status:<10} {speed_str:<15}")

    if any(success for _, _, _, success, _ in results):
        best = max((r for r in results if r[3]), key=lambda x: x[4])
        print(f"\nBest config: page_batch={best[0]}, det_batch={best[1]}, rec_batch={best[2]}")
        print(f"Speed: {best[4]:.3f} pages/sec ({60/best[4]:.1f} sec/page)")

        # Extrapolate to full dataset
        total_pages = 26427 * 12  # Approximate
        hours = (total_pages / best[4]) / 3600
        days = hours / 24
        print(f"\nEstimated time for full archive (~317k pages): {hours:.0f} hours = {days:.1f} days")


if __name__ == "__main__":
    main()
