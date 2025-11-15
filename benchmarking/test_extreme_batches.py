#!/usr/bin/env python3
"""
Test EXTREME batch sizes to find the absolute limit of RTX 5070.

Previous results showed VRAM plateaus at 5.15 GB regardless of batch size.
Testing 48, 64, 96, and 128 pages to find where we finally hit limits.
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

    # Load English sample
    pdf_path = Path("pdf_samples/english/TKP_2009_01_08.pdf")

    # Load N pages (cycle through if PDF doesn't have enough)
    all_images, all_names = load_from_file(pdf_path)

    # Repeat pages to get desired batch size
    images = []
    names = []
    for i in range(num_pages):
        idx = i % len(all_images)
        images.append(all_images[idx])
        names.append(f"{all_names[idx]}_repeat{i}")

    print(f"Processing {len(images)} pages (cycling through {len(all_images)} source pages)")

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
    # Test EXTREME batch sizes
    batch_sizes = [48, 64, 96, 128]

    results = []

    print("\n" + "="*60)
    print("RTX 5070 EXTREME Batch Test - Finding the Limit!")
    print("Testing batch sizes:", batch_sizes)
    print("="*60)

    for batch_size in batch_sizes:
        result = test_batch_size(batch_size)
        results.append(result)

        # If we hit OOM, stop testing larger batches
        if not result['success']:
            print(f"\nHit the limit at batch size {batch_size}!")
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
        print(f"\n🚀 BEST PERFORMANCE: {best['batch_size']} pages at {best['time_per_page']:.1f}s/page")
        print(f"   Throughput: {best['pages_per_sec']:.2f} pages/sec")
        print(f"   Peak memory: {best['peak_memory_gb']:.2f} GB / 12 GB ({best['peak_memory_gb']/12*100:.1f}%)")

        # Calculate full archive timeline
        full_archive_pages = 317000
        total_seconds = full_archive_pages * best['time_per_page']
        total_hours = total_seconds / 3600
        total_days = total_hours / 24

        print(f"\n📊 FULL ARCHIVE ESTIMATE:")
        print(f"   Total time: {total_seconds:.0f}s = {total_hours:.1f} hours = {total_days:.1f} days")
        print(f"   At 16 hrs/day: {total_hours/16:.1f} days")

    failed_results = [r for r in results if not r['success']]
    if failed_results:
        print("\n\n❌ Failed batch sizes:")
        for r in failed_results:
            print(f"  {r['batch_size']} pages: {r.get('error', 'Unknown error')}")
            print(f"    Memory at failure: {r['peak_memory_gb']:.2f} GB")
