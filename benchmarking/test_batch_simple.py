#!/usr/bin/env python
"""
Simple batch test - process multiple pages in ONE call.
"""

import time
import torch
from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames


def test_batch_size(num_pages, det_batch=12, rec_batch=256):
    """Process N pages in a single batch."""
    print(f"\n{'='*60}")
    print(f"Testing: {num_pages} pages in ONE batch")
    print(f"Detection batch: {det_batch}, Recognition batch: {rec_batch}")
    print(f"{'='*60}\n")

    pdf_path = "pdf_samples/english/TKP_2009_01_08.pdf"

    # Load pages
    images, _ = load_from_file(pdf_path)
    images = images[:num_pages]

    # Initialize
    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    initial_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Initial GPU memory: {initial_mem:.2f} GB")

    try:
        start = time.time()

        # Process ALL pages in ONE call
        predictions = rec_predictor(
            images,
            task_names=[TaskNames.ocr_with_boxes] * len(images),
            det_predictor=det_predictor,
            math_mode=False,
            detection_batch_size=det_batch,
            recognition_batch_size=rec_batch
        )

        elapsed = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        total_lines = sum(len(p.text_lines) for p in predictions)

        print(f"\nSUCCESS!")
        print(f"Time: {elapsed:.1f}s ({num_pages/elapsed:.2f} pages/sec)")
        print(f"Per page: {elapsed/num_pages:.1f}s")
        print(f"Peak GPU memory: {peak_mem:.2f} GB")
        print(f"Total lines: {total_lines}")

        return True, num_pages/elapsed, peak_mem

    except torch.cuda.OutOfMemoryError as e:
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nOOM at {peak_mem:.2f} GB peak memory")
        return False, 0, peak_mem

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)[:100]}")
        return False, 0, 0


# Test progressively larger batches
for num_pages in [1, 2, 3, 4, 6, 8]:
    success, speed, mem = test_batch_size(num_pages)

    if not success:
        print(f"\nMax batch size before OOM: {num_pages-1} pages")
        break

    time.sleep(3)  # Cool down between tests

    # If we're using <4GB, we can try more aggressive settings
    if mem < 4.0 and num_pages >= 4:
        print("\n\nMemory usage is low, trying more aggressive batch sizes...\n")
        test_batch_size(num_pages, det_batch=18, rec_batch=512)
