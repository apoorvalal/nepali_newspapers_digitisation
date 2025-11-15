# Nepali Newspapers OCR and Embedding Project

## Overview
Building a modern OCR and semantic search pipeline for a historical archive of Nepali newspapers (2007-2017) containing 26,427 PDFs across 31 newspaper publications in both Nepali and English.

## Data Source
- **Location**: `/media/alal/LAL_DATA/Newspapers/` (external drive mounted via symlink `newspapers_archive`)
- **Size**: 312GB
- **Total PDFs**: 26,427 files
- **Estimated pages**: ~317,000 pages (avg 12 pages/PDF)
- **Languages**: Nepali (Devanagari script) and English
- **Publications include**:
  - The Kathmandu Post (English)
  - Kantipur (Nepali)
  - The Himalayan Times (English)
  - Gorkhapatra (Nepali)
  - And 27 other publications

## Current Status - IN PROGRESS (RTX 5070 BENCHMARKING COMPLETE)

### Environment Setup
- **Working Directory**: `~/tmp/scratch_data/nepali_newspapers`
- **Previous GPU**: NVIDIA GeForce GTX 1070 (8GB VRAM, CUDA 12.2 driver) - BASELINE
- **Current GPU**: NVIDIA RTX 5070 (12GB VRAM, Blackwell architecture with Tensor Cores) - ACTIVE
- **Python Environment**: uv venv created at `.venv`
- **Installed**:
  - PyTorch 2.9.1 with CUDA 12.8 support (upgraded for RTX 5070 compatibility)
  - Surya OCR (latest version)
  - pdftk (for PDF manipulation)

### Sample Data
- English PDFs: `pdf_samples/english/` (5 samples from The Kathmandu Post)
- Nepali PDFs: `pdf_samples/nepali/` (4 samples from Kantipur)

## Phase 1 Progress: OCR Testing & Validation

### ✅ Completed Tasks

1. **Surya OCR Installation**
   - Successfully installed with CUDA support
   - Models auto-downloaded (~1.34GB for recognition model)
   - Verified GPU acceleration working

2. **OCR Quality Testing**
   - Tested on English newspaper (The Kathmandu Post, 2009-01-08)
     - 458 text lines detected on page 1
     - 11,890 characters extracted
     - Quality: Excellent - correctly captured masthead, dates, prices

   - Tested on Nepali newspaper (Kantipur, 2009-01-05)
     - 446 text lines detected on page 1
     - 11,502 characters extracted
     - Quality: Excellent - Devanagari script recognized accurately
     - Sample text verified: "भारत-पाक सकट", "मुम्बई हमलाका दोषी बुकाउन चेतावनी"

3. **Performance Benchmarking on GTX 1070**
   - Single page processing: 200-240 seconds per page
   - Batch processing (2 pages): 223 seconds per page (marginal improvement)
   - Peak GPU memory usage: 5.14 GB (with conservative batch sizes)
   - Bottleneck identified: Detection step (no tensor cores on GTX 1070)

   **Current estimate for full archive**: 550+ days of continuous processing ❌

### 📊 Performance Analysis

**Memory Usage Breakdown:**
- Initial model loading: 1.41 GB
- Detection phase: ~2-3 GB per batch
- Recognition phase: ~1-2 GB per batch
- Peak usage: 5.14 GB (batch size 2, detection_batch=12, recognition_batch=256)

**Why GTX 1070 is inadequate:**
- No Tensor Cores (runs at FP32 instead of optimized FP16/INT8)
- Limited VRAM prevents aggressive batching
- Memory bandwidth constraints for transformer models
- **Result**: ~4-6 minutes per page, impractical for 317k pages

### 🚀 RTX 5070 Actual Performance Results

**Hardware Comparison:**

| Spec | GTX 1070 (Pascal, 2016) | RTX 5070 (Blackwell, 2025) |
|------|------------------------|----------------------------|
| VRAM | 8GB | 12GB (+50%) |
| Tensor Cores | None | 5th gen Tensor Cores |
| FP32 Performance | 6.5 TFLOPS | ~35 TFLOPS |
| Tensor Performance | N/A | 70-80 TFLOPS (FP16) |
| Memory Bandwidth | 256 GB/s | 450-500 GB/s (~2x) |
| CUDA Compute | 6.1 | 12.0 (Blackwell) |

**Single Page Test Results (benchmarking/test_single_page.py):**

| Language | GTX 1070 | RTX 5070 | Speedup | Lines | Chars |
|----------|----------|----------|---------|-------|-------|
| English  | 203.2s   | 11.8s    | **17.2x** | 458   | 11,882 |
| Nepali   | 142.1s   | 9.2s     | **15.5x** | 446   | 11,494 |

**Batch Optimization Results (benchmarking/test_batch_simple.py):**

| Batch Size | Time/Page | Throughput | Peak Memory | Utilization |
|------------|-----------|------------|-------------|-------------|
| 1 page     | 12.2s     | 0.08 p/s   | 5.14 GB     | 43%         |
| 2 pages    | 11.8s     | 0.08 p/s   | 5.15 GB     | 43%         |
| 3 pages    | 12.2s     | 0.08 p/s   | 5.15 GB     | 43%         |
| 4 pages    | 12.4s     | 0.08 p/s   | 5.15 GB     | 43%         |
| 6 pages    | 12.1s     | 0.08 p/s   | 5.15 GB     | 43%         |
| **8 pages** | **10.7s** | **0.09 p/s** | **5.15 GB** | **43%** |

**Key Findings:**
- **Actual Speedup**: 13-19x (exceeded 8-12x projection!)
- **Memory Usage**: Only ~5.15 GB peak regardless of batch size
- **Massive Headroom**: Using less than 50% of 12GB VRAM
- **Best Performance**: 8-page batches at 10.7s/page
- **Conclusion**: Can push much larger batch sizes (12, 16, 24, 32 pages testing)

**Revised Processing Timeline** (based on 10.7s/page):
- Total pages: ~317,000
- Total time: 3,392,000 seconds = 943 hours
- **Continuous**: ~39 days
- **16 hours/day**: ~59 days (2 months)

**Note**: Further optimization with larger batches expected to reduce timeline significantly.

### 📁 Benchmarking Scripts (benchmarking/)

All scripts tested and validated on RTX 5070:

1. **`test_single_page.py`** - Quick single-page OCR validation ✅
   - Tests first page of English and Nepali samples
   - Verifies OCR quality and model loading
   - **RTX 5070 Runtime**: ~21 seconds (2 pages)
   - **Results**: 11.8s English, 9.2s Nepali

2. **`test_batch_simple.py`** - Initial batch optimization (1-8 pages) ✅
   - Tests batch sizes from 1, 2, 3, 4, 6, 8 pages
   - Reports memory usage and throughput for each
   - **Best result**: 8 pages at 10.7s/page, 5.15 GB peak

3. **`test_large_batches.py`** - Large batch optimization (12-32 pages) ⏳
   - Tests batch sizes: 12, 16, 24, 32 pages
   - Pushes GPU to find maximum throughput
   - **Currently running** - exploiting VRAM headroom

4. **`test_batch_optimization.py`** - Detailed tuning parameters
   - Fine-tunes detection and recognition batch sizes
   - Comprehensive parameter sweep

5. **`test_surya_ocr.py`** - Full multi-page PDF processing
   - Processes entire PDFs with configurable batching
   - Includes progress tracking and error handling
   - Saves results as JSON with bounding boxes

### 🗂️ Output Structure

OCR results saved as JSON with structure:
```json
{
  "text_lines": [
    {
      "text": "extracted text",
      "bbox": [x1, y1, x2, y2],
      "polygon": [[x1,y1], [x2,y2], ...],
      "confidence": 0.95
    }
  ],
  "page_number": 1,
  "processing_time": 203.5
}
```

Output directories:
- `ocr_output/quick_test/` - Single page test results
- `ocr_output/parallel/` - Parallel processing results (future)

## Technical Stack

### OCR
- **Surya OCR**: https://github.com/datalab-to/surya
  - Supports 90+ languages including Nepali (Devanagari)
  - GPU-accelerated transformer-based model
  - Better than Tesseract for complex layouts and multilingual text
  - Provides bounding boxes and confidence scores

### Embedding & Search (Phase 3 - Not Started)
Options to evaluate:
- Sentence Transformers (multilingual models like `paraphrase-multilingual-mpnet-base-v2`)
- OpenAI embeddings (if API access available)
- Local models optimized for Nepali/English
- Consider multilingual-e5 or labse for cross-lingual search

### Vector Database (Phase 3 - Not Started)
Options:
- ChromaDB (lightweight, embedded, good for prototyping)
- FAISS (fast, good for large scale, no server needed)
- Qdrant (feature-rich, production-ready, has filtering)

## Implementation Roadmap

### Phase 1: OCR Pipeline ✅ (COMPLETE)
- [x] Install Surya OCR with GPU support
- [x] Test on English sample PDFs
- [x] Test on Nepali sample PDFs
- [x] Verify Devanagari script accuracy
- [x] Benchmark performance on GTX 1070 (baseline)
- [x] Identify bottlenecks and optimization opportunities
- [x] Create test scripts and documentation
- [x] Upgrade to RTX 5070 with PyTorch CUDA 12.8
- [x] Re-benchmark on RTX 5070 (17.2x speedup achieved!)
- [x] Initial batch optimization (1-8 pages)
- [ ] **IN PROGRESS**: Large batch optimization (12-32 pages)
- [ ] **NEXT**: Design markdown output format
- [ ] **NEXT**: Organize repository structure for git

### Phase 2: Production Batch Processing (NEXT)
- [x] Determine initial optimal batch configuration (8 pages)
- [ ] **IN PROGRESS**: Test larger batches (12-32 pages) to maximize GPU
- [ ] Finalize optimal batch size based on throughput/memory tradeoff
- [ ] Create production processing script (code/) with:
  - Resume capability (checkpoint every N PDFs)
  - Progress tracking (tqdm, logging)
  - Error handling (corrupted PDFs, OOM recovery)
  - Publication-wise organization
  - Metadata extraction (date, publication, page count)
- [ ] Set up monitoring (GPU utilization, throughput tracking)
- [ ] Process pilot batch - one publication (~1000 pages) for validation
- [ ] Run full batch processing (~39 days estimated at 10.7s/page, likely faster with larger batches)

### Phase 3: Embedding & Search
1. Evaluate embedding models for Nepali+English
2. Process OCR output into chunks/articles
3. Generate embeddings (batch processing)
4. Set up vector database
5. Build search interface (CLI or simple web UI)
6. Test cross-lingual queries

### Phase 4: Analysis & Queries
1. Build query tools for temporal analysis
2. Topic extraction across publications
3. Cross-lingual topic modeling
4. Trend analysis over 2007-2017 period

## GPU Configuration Notes

### For GTX 1070 (Current - INADEQUATE)
```python
# Conservative settings to avoid OOM
detection_batch_size = 6-12
recognition_batch_size = 128-256
page_batch_size = 1-2
```

### For RTX 5070 (Current - VALIDATED)
```python
# Based on actual benchmarking results:
detection_batch_size = 12  # Working well, room to increase
recognition_batch_size = 256  # Working well, room to increase
page_batch_size = 8  # Current best at 10.7s/page

# Testing larger configurations:
page_batch_size = 12, 16, 24, 32  # Exploiting 57% unused VRAM
# Memory usage remarkably low - only 5.15 GB of 12 GB used
```

### Environment Variable for Memory Management
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Next Steps (Current Session)

1. ✅ **Re-run benchmarks** - Completed with 8-page batching
2. ✅ **Validate speedup** - Achieved 13-19x improvement (exceeded expectations!)
3. ⏳ **Test larger batches** - Currently running 12, 16, 24, 32 page tests
4. **Finalize batch configuration** - Select optimal based on large batch results
5. **Design output format** - Markdown or JSON for extracted text
6. **Build production script** - Implement resume, error handling, progress tracking
7. **Process pilot batch** - One publication (~1000 pages) for validation
8. **Launch full processing** - ~39 days (optimistic with larger batches)

## Known Issues & Considerations

- **PDF variations**: Some newspapers may have different layouts/quality
- **Corrupted files**: Legacy code showed some failures - need error handling
- **Date extraction**: Need to parse dates from filenames or text
- **Storage**: 317k pages of JSON will be substantial - consider compression
- **Selective processing**: May want to prioritize certain publications/years
- **Quality validation**: Random sampling to verify OCR accuracy

## Repository Structure

### benchmarking/ - Performance Testing
- `test_single_page.py` - Quick single-page validation ✅
- `test_batch_simple.py` - Batch optimization 1-8 pages ✅
- `test_large_batches.py` - Large batch optimization 12-32 pages ⏳
- `test_batch_optimization.py` - Detailed parameter tuning
- `test_surya_ocr.py` - Full PDF processing test
- `batch_test_results.txt` - Historical test results

### code/ - Production Processing
- `parallel_ocr.py` - Experimental parallel processing (legacy)
- Production script to be developed

### planning/ - Documentation
- `PROJECT_PLAN.md` - This file - comprehensive roadmap
- `SCRIPTS_README.md` - Benchmarking scripts documentation
- `RTX5070_SETUP_CHECKLIST.md` - GPU setup guide
- `README.md` - Quick overview (legacy)

### Data Directories
- `pdf_samples/english/` - English newspaper samples (The Kathmandu Post)
- `pdf_samples/nepali/` - Nepali newspaper samples (Kantipur)
- `ocr_output/` - OCR results (gitignored)
- `newspapers_archive/` - Symlink to full archive (gitignored)
- `.venv/` - Python virtual environment (gitignored)

### Legacy Code (old_nb/)
- Jupyter notebooks with old textract-based approach
- NLTK analysis and n-gram visualizations

## References

- Surya OCR: https://github.com/datalab-to/surya
- PyTorch CUDA docs: https://pytorch.org/docs/stable/notes/cuda.html
- RTX 5070 specs: (TBD - will update after official release)

---

**Last Updated**: 2025-11-14
**Status**: Active - RTX 5070 benchmarking complete, testing large batches
**Current Focus**: Maximize GPU utilization with 12-32 page batching, then build production script
**Next Milestone**: Launch pilot processing run on single publication
