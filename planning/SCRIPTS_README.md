# Benchmarking Scripts Documentation

Test scripts for Surya OCR performance testing and optimization on RTX 5070.

Location: `benchmarking/`

## Prerequisites

```bash
# Activate the uv virtual environment
source .venv/bin/activate

# Set memory optimization (required for RTX 5070)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Scripts Overview

### 1. `test_single_page.py` - Quick Validation ✅

**Purpose**: Fast sanity check verifying OCR quality on English and Nepali.

**What it does**:
- Tests page 1 of The Kathmandu Post (English sample)
- Tests page 1 of Kantipur (Nepali sample)
- Prints first 10 lines of extracted text
- Saves JSON results to `ocr_output/quick_test/`

**Usage**:
```bash
python benchmarking/test_single_page.py
```

**RTX 5070 Results**:
- Runtime: 21 seconds (2 pages)
- English: 11.8s, 458 lines, 11,882 chars
- Nepali: 9.2s, 446 lines, 11,494 chars
- **Speedup vs GTX 1070**: 17.2x (English), 15.5x (Nepali)

**When to use**: First test after GPU installation to verify everything works.

---

### 2. `test_batch_simple.py` - Batch Optimization ✅

**Purpose**: Find optimal batch size by testing 1, 2, 3, 4, 6, 8 pages.

**What it does**:
- Processes progressively larger batches from a single PDF
- Reports throughput (pages/sec) and peak memory for each
- Identifies best configuration
- Stops if OOM occurs

**Usage**:
```bash
python benchmarking/test_batch_simple.py
```

**Runtime**: ~10-15 minutes

**RTX 5070 Results**:
| Batch Size | Time/Page | Throughput | Peak Memory |
|------------|-----------|------------|-------------|
| 1 page     | 12.2s     | 0.08 p/s   | 5.14 GB     |
| 2 pages    | 11.8s     | 0.08 p/s   | 5.15 GB     |
| 4 pages    | 12.4s     | 0.08 p/s   | 5.15 GB     |
| 6 pages    | 12.1s     | 0.08 p/s   | 5.15 GB     |
| **8 pages** | **10.7s** | **0.09 p/s** | **5.15 GB** |

**Key Finding**: Memory usage plateaus at ~5.15 GB regardless of batch size - massive headroom for larger batches!

**When to use**: First optimization test to establish baseline batch configuration.

---

### 3. `test_large_batches.py` - Maximum Throughput ⏳

**Purpose**: Push GPU to limits with 12, 16, 24, 32 page batches.

**What it does**:
- Tests large batch sizes to maximize GPU utilization
- Exploits 57% unused VRAM from initial tests
- Stops on first OOM error
- Reports optimal configuration

**Usage**:
```bash
python benchmarking/test_large_batches.py
```

**Runtime**: ~20-30 minutes

**Status**: Currently running - testing 12, 16, 24, 32 pages

**When to use**: After test_batch_simple.py shows significant memory headroom.

---

### 4. `test_batch_optimization.py` - Fine Tuning

**Purpose**: Comprehensive testing of detection and recognition batch parameters.

**What it does**:
- Tests combinations of page_batch, detection_batch, recognition_batch
- Provides detailed metrics for each configuration
- Advanced tuning beyond simple batch sizing

**Usage**:
```bash
python benchmarking/test_batch_optimization.py
```

**When to use**: Advanced optimization after establishing baseline with simpler tests.

**Note**: Time-consuming - run test_batch_simple.py first.

---

### 5. `test_surya_ocr.py` - Full PDF Processing

**Purpose**: Process complete PDFs with configurable batching for validation.

**What it does**:
- Processes all pages in specified PDFs
- Uses optimized batch sizes from benchmark tests
- Progress tracking with tqdm
- Saves complete JSON results
- Error handling

**Usage**:
```bash
# Configure batch sizes in script based on optimization results
python benchmarking/test_surya_ocr.py
```

**Configuration** (edit in script):
```python
# Based on RTX 5070 results:
page_batch_size = 8  # or higher from test_large_batches.py
detection_batch_size = 12
recognition_batch_size = 256
```

**When to use**: Validate full PDF processing before production run.

---

## Quick Start Guide (RTX 5070)

```bash
# 1. Activate environment
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Quick validation (21 seconds)
python benchmarking/test_single_page.py

# 3. Find optimal batch size (10-15 min)
python benchmarking/test_batch_simple.py

# 4. Test larger batches if memory headroom exists (20-30 min)
python benchmarking/test_large_batches.py

# 5. Use optimal config in production script
```

## RTX 5070 Performance Summary

| Metric | GTX 1070 | RTX 5070 | Improvement |
|--------|----------|----------|-------------|
| Page processing time | 200-240s | 10.7s | **19x** |
| Optimal batch size | 1-2 pages | 8+ pages | 4-8x |
| Peak memory usage | 5.15 GB | 5.15 GB | Same |
| Detection batch | 12 | 12 | - |
| Recognition batch | 256 | 256 | - |
| **Full archive time** | **550+ days** | **~39 days** | **14x** |

**Key Insights**:
- Memory usage unchanged despite 19x speedup - bottleneck is compute, not memory
- Only using 43% of available 12 GB VRAM with 8-page batches
- Larger batches (12-32 pages) likely to further improve throughput
- Tensor Cores providing massive acceleration for transformer inference

## Output Format

All scripts save OCR results as JSON:

```json
{
  "text_lines": [
    {
      "text": "NEPAL'S LARGEST SELLING ENGLISH DAILY",
      "bbox": [90, 2, 364, 13],
      "polygon": [[90, 2], [364, 2], [364, 13], [90, 13]],
      "confidence": 0.98
    }
  ],
  "page_number": 1,
  "processing_time": 11.8,
  "source_pdf": "path/to/file.pdf"
}
```

## Troubleshooting

### Out of Memory Errors

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions** (in order of impact):
1. Reduce `page_batch_size` (number of pages processed together)
2. Reduce `detection_batch_size` (internal detection batching)
3. Reduce `recognition_batch_size` (internal recognition batching)
4. Ensure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set
5. Check for other GPU processes: `nvidia-smi`

### Slow Performance on RTX 5070

If not achieving ~10-12s/page:

1. **Verify CUDA 12.8**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   # Should print: 12.8
   ```

2. **Check GPU detection**:
   ```bash
   python -c "import torch; print(torch.cuda.get_device_name())"
   # Should print: NVIDIA GeForce RTX 5070
   ```

3. **Monitor GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi
   # Should show high GPU-Util (80-100%) during processing
   ```

4. **Check thermal throttling**:
   - Temperature should be <85°C
   - Power draw should be near TDP limit

### Import Errors

```
ModuleNotFoundError: No module named 'surya.ocr'
```

**Solution**: Use correct imports:
```python
from surya.input.load import load_from_file
from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames
```

### Model Download Issues

First run downloads ~1.34GB of models to `~/.cache/datalab/models/`

If download fails:
- Check internet connection
- Verify disk space
- Models should auto-download on first predictor initialization

## Next Steps

After benchmarking completes:

1. **Finalize batch configuration** based on test_large_batches.py results
2. **Create production script** in `code/` with optimal settings:
   - Resume capability (checkpoints)
   - Progress tracking (tqdm with ETA)
   - Error handling (corrupted PDFs, OOM recovery)
   - Publication-wise organization
   - Metadata extraction (date, publication, page count)
3. **Run pilot batch** - one publication (~1000 pages)
4. **Launch full processing** - ~317,000 pages, estimated 39 days (likely faster with optimized batching)

---

**Last Updated**: 2025-11-14
**Status**: RTX 5070 benchmarking complete (1-8 pages), testing large batches (12-32 pages)
**Best Config**: 8 pages, 12 detection, 256 recognition, 10.7s/page, 5.15 GB peak
