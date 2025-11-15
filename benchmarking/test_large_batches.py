#!/usr/bin/env python3
"""
Test larger batch sizes to maximize RTX 5070 utilization.

Previous test (test_batch_simple.py) showed:
- 8 pages: 10.7s/page, 5.15 GB peak memory
- Only using ~43% of available 12 GB VRAM
- Clear headroom for larger batches

This test pushes to 12, 16, 24, and 32 pages to find the sweet spot.
"""

import time
from pathlib import Path
import torch
from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0

def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def test_batch_size(num_pages, det_batch=12, rec_batch=256):
    """Test processing num_pages in a single batch."""

    print(f"\n{'='*60}")
    print(f"Testing: {num_pages} pages in ONE batch")
    print(f"Detection batch: {det_batch}, Recognition batch: {rec_batch}")
    print(f"{'='*60}\n")

    # Load English sample (tends to be slower/more memory intensive)
    pdf_path = Path("pdf_samples/english/TKP_2009_01_08.pdf")

    # Load N pages
    images, names = load_from_file(pdf_path)
    images = images[:num_pages]
    names = names[:num_pages]

    # Initialize models
    foundation_predictor = FoundationPredictor()
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor(foundation_predictor)

    # Reset GPU memory tracking
    reset_gpu_memory()
    initial_mem = get_gpu_memory()
    print(f"Initial GPU memory: {initial_mem:.2f} GB\n")

    try:
        # Run OCR on all pages in one batch
        start_time = time.time()

        predictions = rec_predictor(
            images,
            task_names=[TaskNames.ocr_with_boxes] * len(images),
            det_predictor=det_predictor,
            math_mode=False,
            detection_batch_size=det_batch,
            recognition_batch_size=rec_batch
        )

        end_time = time.time()
        elapsed = end_time - start_time

        # Get stats
        peak_mem = get_gpu_memory()
        total_lines = sum(len(pred.text_lines) for pred in predictions)
        pages_per_sec = num_pages / elapsed
        time_per_page = elapsed / num_pages

        print(f"\nSUCCESS!")
        print(f"Time: {elapsed:.1f}s ({pages_per_sec:.2f} pages/sec)")
        print(f"Per page: {time_per_page:.1f}s")
        print(f"Peak GPU memory: {peak_mem:.2f} GB")
        print(f"Total lines: {total_lines}")

        return {
            'batch_size': num_pages,
            'total_time': elapsed,
            'time_per_page': time_per_page,
            'pages_per_sec': pages_per_sec,
            'peak_memory_gb': peak_mem,
            'total_lines': total_lines,
            'success': True
        }

    except Exception as e:
        print(f"\nFAILED!")
        print(f"Error: {str(e)}")
        peak_mem = get_gpu_memory()
        print(f"Peak GPU memory at failure: {peak_mem:.2f} GB")

        return {
            'batch_size': num_pages,
            'success': False,
            'error': str(e),
            'peak_memory_gb': peak_mem
        }

    finally:
        # Cleanup
        reset_gpu_memory()

if __name__ == "__main__":
    # Test progressively larger batches
    batch_sizes = [12, 16, 24, 32]

    results = []

    print("\n" + "="*60)
    print("RTX 5070 Large Batch Optimization Test")
    print("Testing batch sizes:", batch_sizes)
    print("="*60)

    for batch_size in batch_sizes:
        result = test_batch_size(batch_size)
        results.append(result)

        # If we hit OOM, stop testing larger batches
        if not result['success']:
            print(f"\nStopping tests - batch size {batch_size} failed")
            break

        # Brief pause between tests
        time.sleep(2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful_results = [r for r in results if r['success']]

    if successful_results:
        print("\nSuccessful batch sizes:")
        print(f"{'Batch Size':<12} {'Time/Page':<12} {'Throughput':<15} {'Peak Memory':<15}")
        print("-" * 60)

        for r in successful_results:
            print(f"{r['batch_size']:<12} {r['time_per_page']:<12.1f} "
                  f"{r['pages_per_sec']:<15.2f} {r['peak_memory_gb']:<15.2f} GB")

        # Find best performer
        best = min(successful_results, key=lambda x: x['time_per_page'])
        print(f"\nBest performance: {best['batch_size']} pages at {best['time_per_page']:.1f}s/page")
        print(f"Peak memory usage: {best['peak_memory_gb']:.2f} GB / 12 GB ({best['peak_memory_gb']/12*100:.1f}%)")

    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("\n\nFailed batch sizes:")
        for r in failed_results:
            print(f"  {r['batch_size']} pages: {r['error']}")
