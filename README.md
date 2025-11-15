# Nepali Newspapers OCR Project

Large-scale OCR processing of historical Nepali newspaper archive (2007-2017) using Surya OCR on RTX 5070.

## Project Overview

**Goal**: Extract searchable text from 26,427 Nepali/English newspaper PDFs (~317,000 pages)

**OCR Engine**: [Surya OCR](https://github.com/VikParuchuri/surya) - Transformer-based multilingual OCR
- Supports 90+ languages including Nepali (Devanagari script)
- High accuracy on complex scripts and mixed-language documents
- GPU-accelerated with batch processing

**Hardware**: NVIDIA RTX 5070 (12 GB VRAM, Blackwell architecture, CUDA 12.8)

## Repository Structure

```
nepali_newspapers/
├── benchmarking/          # Performance testing and optimization scripts
│   ├── test_single_page.py       # Quick single-page validation
│   ├── test_batch_simple.py      # Initial batch optimization (1-8 pages)
│   ├── test_large_batches.py     # Large batch optimization (12-32 pages)
│   ├── test_batch_optimization.py # Detailed batch tuning
│   └── test_surya_ocr.py         # Full PDF processing test
│
├── code/                  # Production processing code
│   ├── process_newspaper_archive.py  # Main production OCR script
│   ├── count_publications.py         # Publication analysis tool
│   └── parallel_ocr.py               # Experimental parallel processing (legacy)
│
├── planning/              # Project documentation
│   ├── PROJECT_PLAN.md           # Comprehensive project roadmap
│   ├── SCRIPTS_README.md         # Benchmarking scripts documentation
│   ├── RTX5070_SETUP_CHECKLIST.md # GPU setup guide
│   └── README.md                 # Quick overview (legacy)
│
├── pdf_samples/           # Test PDFs
│   ├── english_sample/           # English newspaper samples
│   └── nepali_sample/            # Nepali newspaper samples
│
├── ocr_output/            # OCR results (gitignored)
├── newspapers_archive/    # Symlink to full archive (gitignored)
└── .venv/                 # Python virtual environment (gitignored)
```

## Performance Results (RTX 5070)

### Latest Benchmarks

**Single Page Test** (test_single_page.py):
- English: 11.8s (458 lines, 11,882 chars)
- Nepali: 9.2s (446 lines, 11,494 chars)

**Batch Optimization** (test_batch_simple.py, test_large_batches.py, test_extreme_batches.py):
| Batch Size | Time/Page | Throughput | Peak Memory | Notes |
|------------|-----------|------------|-------------|-------|
| 1 page     | 12.2s     | 0.08 p/s   | 5.14 GB     | Baseline |
| 8 pages    | 10.7s     | 0.09 p/s   | 5.15 GB     | Good |
| 16 pages   | 7.0s      | 0.14 p/s   | 5.15 GB     | Better |
| 24 pages   | 4.7s      | 0.21 p/s   | 5.15 GB     | Great |
| **32 pages** | **3.4s** | **0.29 p/s** | **5.15 GB** | **OPTIMAL** |
| 48 pages   | 8.9s      | 0.11 p/s   | 5.15 GB     | Degradation |
| 64 pages   | 9.0s      | 0.11 p/s   | 5.15 GB     | Worse |

**Key Finding**: VRAM usage plateaus at 5.15 GB regardless of batch size. Performance peaks at 32 pages, then degrades due to pipeline overhead.

### RTX 5070 vs GTX 1070 Comparison

| Metric | GTX 1070 | RTX 5070 (optimal) | Speedup |
|--------|----------|----------|---------|
| Time per page | 203.2s | 3.4s | **59.8x** |
| Batch size | 1-2 pages | 32 pages | 16-32x |
| VRAM usage | 5.15 GB | 5.15 GB | Same |
| VRAM available | 8 GB | 12 GB | 1.5x |
| Tensor Cores | None | 5th Gen | - |
| Full archive | 550+ days | 12.5 days | **44x faster** |

**Key Finding**: With optimal 32-page batching, RTX 5070 achieves 60x per-page speedup and 44x total throughput improvement!

## Timeline Estimates

### Full Archive (26,427 PDFs, ~317,000 pages)
Based on 3.4s/page with optimal 32-page batching:
- **Continuous processing**: 12.5 days
- **16 hours/day**: 18.7 days

### Major Publications Subset (16,884 PDFs, ~202,600 pages)
Kantipur, Kathmandu Post, Nagarik, Republica, Annapurna Post, Himalayan Times:
- **Continuous processing**: 8.0 days
- **16 hours/day**: 12.0 days

**Recommended approach**: Focus on major publications first for highest quality/impact.

## Quick Start

### Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify PyTorch CUDA 12.8
python -c "import torch; print(f'CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name()}')"

# Expected: CUDA: 12.8, GPU: NVIDIA GeForce RTX 5070
```

### Count Publications

```bash
# Analyze major publications
python code/count_publications.py
```

### Process Newspapers (Production)

```bash
# Process a single publication
python code/process_newspaper_archive.py "newspapers_archive/Kantipur"

# Process all major publications
for pub in "Kantipur" "The Kathmandu Post" "Nagarik" "Republica" "Annapurna Post" "The Himalayan Times"; do
    python code/process_newspaper_archive.py "newspapers_archive/$pub"
done
```

### Run Benchmarks (Optional)

```bash
# Single page validation
python benchmarking/test_single_page.py

# Batch optimization tests
python benchmarking/test_batch_simple.py
python benchmarking/test_large_batches.py
python benchmarking/test_extreme_batches.py
```

## Project Status

- [x] **Phase 1**: GTX 1070 baseline testing
- [x] **Phase 2**: RTX 5070 upgrade and validation
- [x] **Phase 3**: Batch optimization (1-32 pages)
- [x] **Phase 4**: Extreme batch testing (48-128 pages)
- [x] **Phase 5**: Production script development
- [ ] **Phase 6**: Pilot run (single publication)
- [ ] **Phase 7**: Major publications processing (6 pubs, 8 days)
- [ ] **Phase 8**: Full archive processing (optional)

**Next**: Run pilot on one publication to validate production pipeline.

See [planning/PROJECT_PLAN.md](planning/PROJECT_PLAN.md) for detailed roadmap.

## Technical Details

**Dependencies**:
- Python 3.11+
- PyTorch 2.9.1 with CUDA 12.8
- Surya OCR (latest)
- uv (package management)

**GPU Configuration**:
- CUDA compute capability: 12.0 (Blackwell)
- Memory allocator: expandable_segments enabled
- Batch processing: Detection (12), Recognition (256)

## Data

**Archive**: 26,427 PDFs from major Nepali newspapers (2007-2017)
- Mix of Nepali (Devanagari) and English text
- Variable page counts (1-40 pages per PDF)
- Average ~12 pages per PDF
- Total estimated: ~317,000 pages

**Samples**: Located in `pdf_samples/`
- English: Kathmandu Post, My Republica
- Nepali: Kantipur, Nagarik

## Notes

- Memory usage remarkably low (~5.15 GB) even with 8-page batches
- Significant VRAM headroom suggests larger batches feasible
- OCR quality excellent for both English and Devanagari scripts
- Processing time dominated by recognition, not detection

## License

Project-specific license TBD. Surya OCR is Apache 2.0.

## Contact

Internal project documentation.
