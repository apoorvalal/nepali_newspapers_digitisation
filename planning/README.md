# Nepali Newspapers OCR Project

Historical newspaper archive OCR and semantic search pipeline for 26,427 Nepali and English PDFs (2007-2017).

## Project Status: PAUSED - Awaiting RTX 5070 GPU

**Current GPU**: GTX 1070 (8GB) - Inadequate for large-scale processing  
**Upgrade**: RTX 5070 (12GB) with Tensor Cores - Expected 8-12x speedup  
**Progress**: Phase 1 complete, OCR validated, benchmarks done

## Quick Start (After RTX 5070 Installation)

```bash
cd ~/tmp/scratch_data/nepali_newspapers
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Verify GPU setup
python test_single_page.py

# 2. Benchmark optimal batch sizes
python test_batch_simple.py

# 3. See RTX5070_SETUP_CHECKLIST.md for full setup
```

## Key Findings

### OCR Quality: Excellent ✅
- **English**: 458 lines, 11,890 chars per page, accurate text capture
- **Nepali**: 446 lines, 11,502 chars per page, Devanagari script perfect

### Current Performance: Unusable ❌
- **GTX 1070**: 200-240 seconds per page
- **Full archive**: 550+ days continuous processing
- **Bottleneck**: No Tensor Cores for transformer inference

### Expected RTX 5070 Performance: Excellent ✅
- **Speedup**: 8-12x faster (tensor cores + memory + bandwidth)
- **Per page**: 20-30 seconds
- **Full archive**: 6-9 days continuous processing

## Documentation

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - Comprehensive project status and roadmap
- **[SCRIPTS_README.md](SCRIPTS_README.md)** - Test script documentation
- **[RTX5070_SETUP_CHECKLIST.md](RTX5070_SETUP_CHECKLIST.md)** - Setup guide for new GPU

## File Structure

```
nepali_newspapers/
├── PROJECT_PLAN.md              # Main project documentation
├── SCRIPTS_README.md            # Script usage guide
├── RTX5070_SETUP_CHECKLIST.md   # GPU setup checklist
├── README.md                    # This file
├── test_single_page.py          # Quick validation test
├── test_batch_simple.py         # Batch optimization test
├── test_surya_ocr.py            # Full PDF processing
├── test_batch_optimization.py   # Detailed tuning
├── parallel_ocr.py              # Experimental parallel processing
├── pdf_samples/
│   ├── english/                 # English newspaper samples
│   └── nepali/                  # Nepali newspaper samples
├── ocr_output/                  # OCR test results
└── .venv/                       # Python virtual environment
```

## Data Overview

- **Location**: `/media/alal/LAL_DATA/Newspapers/` (symlinked as `newspapers_archive`)
- **Size**: 312GB, 26,427 PDFs, ~317,000 pages
- **Languages**: Nepali (Devanagari) and English
- **Date Range**: 2007-2017
- **Publications**: 31 newspapers including Kantipur, The Kathmandu Post, etc.

## Next Session Checklist

When you get the RTX 5070:

1. ✅ Install GPU and drivers
2. ✅ Run `python test_single_page.py` - should take ~30-60 seconds
3. ✅ Run `python test_batch_simple.py` - find optimal batch sizes
4. ✅ Build production script with optimal settings
5. ✅ Process pilot subset (~1000 pages) for validation
6. ✅ Launch full batch processing (~6-9 days runtime)

## Technology Stack

- **OCR**: Surya OCR (transformer-based, 90+ languages)
- **GPU**: PyTorch 2.9.1 with CUDA 12.6
- **Processing**: Python with batch optimization
- **Output**: JSON with text + bounding boxes
- **Future**: Embeddings + vector database for semantic search

---

**Last Updated**: 2025-11-13  
**Status**: Phase 1 complete, ready for RTX 5070  
**Next**: Benchmark → Optimize → Production Processing
