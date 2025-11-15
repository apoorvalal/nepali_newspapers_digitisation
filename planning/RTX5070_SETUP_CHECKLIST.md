# RTX 5070 Setup Checklist

Quick reference for getting back up and running when the new GPU arrives.

## Hardware Installation

- [ ] Physically install RTX 5070
- [ ] Connect power cables (likely 12VHPWR or dual 8-pin)
- [ ] Boot system and verify GPU detected: `lspci | grep -i nvidia`

## CUDA Setup

### Option 1: Fresh CUDA Install for RTX 5070

```bash
# Check current CUDA version
nvcc --version
nvidia-smi

# RTX 5070 (Blackwell) will need CUDA 12.x or newer
# If current CUDA is too old, update:
# Visit: https://developer.nvidia.com/cuda-downloads
# Select Linux -> x86_64 -> Ubuntu -> version -> deb (local)
```

### Option 2: Use Existing CUDA 12.2

Current driver (535.274.02) *should* work with RTX 5070, but may need update.

```bash
# Test current setup
nvidia-smi

# If it shows RTX 5070 with no errors, proceed to PyTorch test
```

## PyTorch Verification

```bash
cd ~/tmp/scratch_data/nepali_newspapers
source .venv/bin/activate

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected output**:
```
PyTorch: 2.9.1+cu126
CUDA available: True
CUDA version: 12.6
GPU: NVIDIA GeForce RTX 5070
```

### If PyTorch Doesn't Detect GPU

May need PyTorch rebuild for Blackwell architecture:

```bash
# Check if newer PyTorch is available
pip list | grep torch

# Update if needed (check pytorch.org for latest CUDA 12.x wheel)
pip install --upgrade torch torchvision torchaudio
```

## Surya OCR Verification

```bash
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Quick test (~30-60 seconds on RTX 5070)
python test_single_page.py
```

**Expected output**:
```
Results for TKP_2009_01_08.pdf - Page 1
Text lines detected: 458
Total characters: 11890

Results for KPUR_2009_01_05.pdf - Page 1
Text lines detected: 446
Total characters: 11502
```

**Expected time**: 30-60 seconds total (vs 6 minutes on GTX 1070)

## Benchmark & Optimize

```bash
# Find optimal batch sizes for RTX 5070
# This will take 20-40 minutes
python test_batch_simple.py
```

**Look for**:
- Max batch size before OOM (expecting 6-8 pages)
- Peak memory usage (should stay under 12GB)
- Pages per second (expecting 0.05-0.08, i.e., 12-20 sec/page)

**Example good output**:
```
Testing: 6 pages in ONE batch
SUCCESS!
Time: 120.0s (0.05 pages/sec)
Per page: 20.0s
Peak GPU memory: 9.5 GB
```

## Update Configuration

Based on benchmark results, update these values in production scripts:

```python
# In your production OCR script:
page_batch_size = X        # From test_batch_simple.py (likely 6-8)
detection_batch_size = Y   # Start with 18-24
recognition_batch_size = Z # Start with 512-1024
```

## Production Test Run

Before processing the full archive:

```bash
# Create production script with optimal settings
# Test on one newspaper (e.g., one day of The Kathmandu Post)
# Verify output quality
# Check processing speed matches benchmarks
```

## Monitoring During Production

```bash
# In one terminal, monitor GPU:
watch -n 1 nvidia-smi

# In another, run production script with logging
# Should see consistent GPU utilization 95-100%
# Temperature should stay under 80-85°C
```

## Expected Performance Gains

| Metric | GTX 1070 | RTX 5070 (Target) | Speedup |
|--------|----------|-------------------|---------|
| Single page | 240s | 20-30s | 8-12x |
| Batch processing | 223s/page | 18-25s/page | 9-12x |
| Peak memory | 5GB | 9-10GB | - |
| Full archive | 550 days | 6-9 days | ~80x |

## Troubleshooting

### GPU Not Detected
```bash
# Check if GPU is visible to system
lspci | grep -i nvidia

# Check driver
nvidia-smi

# If "No devices were found", reinstall driver
# Visit: https://www.nvidia.com/download/index.aspx
```

### Performance Not Meeting Expectations

If still seeing 100+ seconds per page on RTX 5070:

1. **Check GPU utilization**: Should be 95-100% during processing
   ```bash
   nvidia-smi
   ```

2. **Verify Tensor Cores are enabled**: Should auto-enable with FP16
   ```bash
   # Look for FP16/INT8 in model configs
   ```

3. **Check power/thermal throttling**:
   ```bash
   nvidia-smi -q -d PERFORMANCE
   ```

4. **Ensure using CUDA-optimized build**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   # Should show 12.x
   ```

### Memory Issues Even with 12GB

If hitting OOM with reasonable batch sizes:

1. Reduce detection_batch_size first (biggest memory user)
2. Check for memory leaks: `torch.cuda.empty_cache()`
3. Verify no other processes using GPU: `nvidia-smi`

## Next Steps After Validation

Once benchmarks look good:

1. **Design markdown output format** (for storing OCR text)
2. **Build production batch script** with:
   - Progress tracking
   - Checkpointing
   - Error handling
   - Metadata extraction
3. **Test on pilot subset** (~1000 pages)
4. **Launch full production run** (~6-9 days)

## Contact Info / Notes

When resuming with `claude -c`:

```bash
# Location
cd ~/tmp/scratch_data/nepali_newspapers

# Key files
PROJECT_PLAN.md          # Overall project status
SCRIPTS_README.md        # Test script documentation
RTX5070_SETUP_CHECKLIST.md  # This file

# First command after GPU setup
source .venv/bin/activate && python test_single_page.py
```

---

**Status**: Ready for RTX 5070 installation
**Expected Setup Time**: 30-60 minutes (driver + verification + benchmarks)
**Next Session**: Benchmark → Optimize → Production Script → Pilot Run → Full Batch
