#!/usr/bin/env python3
"""
Count PDFs in major publications and estimate processing times.
"""

from pathlib import Path
import sys

MAJOR_PUBLICATIONS = [
    "Kantipur",
    "The Kathmandu Post",
    "Nagarik",
    "Republica",
    "Annapurna Post",
    "The Himalayan Times"
]

AVG_PAGES_PER_PDF = 12  # Conservative estimate
TIME_PER_PAGE = 3.4     # RTX 5070 with 32-page batching


def count_pdfs_in_directory(directory: Path) -> int:
    """Count PDF files in directory recursively."""
    if not directory.exists():
        return 0
    return len(list(directory.rglob("*.pdf")))


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    hours = seconds / 3600
    days = hours / 24

    if days >= 1:
        return f"{days:.1f} days ({hours:.1f} hours)"
    elif hours >= 1:
        return f"{hours:.1f} hours"
    else:
        return f"{seconds/60:.1f} minutes"


def main(archive_path: str = "newspapers_archive"):
    """Count PDFs and estimate processing times for major publications."""

    archive_root = Path(archive_path)

    if not archive_root.exists():
        print(f"Error: Archive directory not found: {archive_root}")
        sys.exit(1)

    print("="*70)
    print("NEPALI NEWSPAPERS - MAJOR PUBLICATIONS ANALYSIS")
    print("="*70)
    print(f"\nArchive: {archive_root.resolve()}")
    print(f"RTX 5070 Performance: {TIME_PER_PAGE}s per page (32-page batching)")
    print(f"Estimated pages per PDF: {AVG_PAGES_PER_PDF}")
    print()

    total_pdfs = 0
    total_pages = 0
    total_seconds = 0

    results = []

    for pub_name in MAJOR_PUBLICATIONS:
        pub_dir = archive_root / pub_name
        pdf_count = count_pdfs_in_directory(pub_dir)

        if pdf_count > 0:
            est_pages = pdf_count * AVG_PAGES_PER_PDF
            est_seconds = est_pages * TIME_PER_PAGE

            results.append({
                'name': pub_name,
                'pdfs': pdf_count,
                'pages': est_pages,
                'seconds': est_seconds,
                'exists': True
            })

            total_pdfs += pdf_count
            total_pages += est_pages
            total_seconds += est_seconds
        else:
            results.append({
                'name': pub_name,
                'pdfs': 0,
                'pages': 0,
                'seconds': 0,
                'exists': False
            })

    # Print results
    print(f"{'Publication':<25} {'PDFs':<10} {'Est. Pages':<12} {'Processing Time':<20}")
    print("-"*70)

    for result in results:
        if result['exists']:
            lang = "(EN)" if "Kathmandu Post" in result['name'] or "Republica" in result['name'] or "Himalayan" in result['name'] else "(NP)"
            time_str = format_time(result['seconds'])
            print(f"{result['name']:<25} {result['pdfs']:<10} {result['pages']:<12} {time_str:<20} {lang}")
        else:
            print(f"{result['name']:<25} {'NOT FOUND':<10}")

    print("-"*70)
    print(f"{'TOTAL':<25} {total_pdfs:<10} {total_pages:<12} {format_time(total_seconds):<20}")
    print()

    if total_pdfs > 0:
        # Processing time estimates
        continuous_days = total_seconds / 86400
        sixteen_hour_days = (total_seconds / 3600) / 16

        print("PROCESSING TIME ESTIMATES:")
        print(f"  Continuous (24/7): {continuous_days:.1f} days")
        print(f"  At 16 hours/day: {sixteen_hour_days:.1f} days")
        print()

        # What's interesting about this subset
        print("WHY THIS SUBSET IS INTERESTING:")
        print()
        print("1. MAJOR PUBLICATIONS - Most authoritative/widely-read newspapers")
        print("   - Highest circulation and influence")
        print("   - Better print quality = better OCR accuracy")
        print()
        print("2. LANGUAGE BALANCE - Mix of Nepali and English")
        print("   - Nepali (Devanagari): Kantipur, Nagarik, Annapurna Post")
        print("   - English (Latin): Kathmandu Post, Republica, Himalayan Times")
        print("   - Enables cross-lingual comparative analysis")
        print()
        print("3. POLITICAL DIVERSITY - Different editorial perspectives")
        print("   - Can analyze same events from multiple viewpoints")
        print("   - Track narrative differences across publications")
        print()
        print("4. TEMPORAL COVERAGE - 2007-2017 period includes:")
        print("   - Constitutional assembly elections (2008)")
        print("   - Federal republic transition (2008)")
        print("   - Political instability period (2008-2015)")
        print("   - 2015 earthquake and aftermath")
        print("   - New constitution (2015)")
        print()
        print("5. RESEARCH POTENTIAL:")
        print("   - Cross-lingual topic modeling")
        print("   - Sentiment analysis across languages")
        print("   - Event detection and tracking")
        print("   - Media framing analysis (English vs Nepali coverage)")
        print("   - Political discourse evolution")
        print()


if __name__ == "__main__":
    archive = sys.argv[1] if len(sys.argv) > 1 else "newspapers_archive"
    main(archive)
