#!/usr/bin/env python
"""
Test script for PRISM Agent 1: PREPROCESSOR PRIME
Tests the preprocessor with actual L4-MOSAIC data from the user's environment.
"""

import sys
import os
from pathlib import Path

# Add the agents directory to the path so we can import agent1_preprocessor
sys.path.append(str(Path(__file__).parent / "agents"))

from agents.agent1_preprocessor import PreprocessorPrime

def main():
    print("Testing PRISM Agent 1: PREPROCESSOR PRIME")
    print("=" * 50)

    # Paths to the actual L4-MOSAIC data the user verified earlier
    base_path = "D:/PRISM_DATA/01_DFSAR"

    config = {
        'cpr_path': f"{base_path}/mpcpspeast/data/derived/20250630/ch2_sar_ndxl_20250630mpcpspeast_d_cpr_xx_fp_xx_xxx.tif",
        'srd_path': f"{base_path}/mpcpspeast/data/derived/20250630/ch2_sar_ndxl_20250630mpcpspeast_d_srd_xx_fp_xx_xxx.tif",
        'trt_path': f"{base_path}/mpcpspeast/data/derived/20250630/ch2_sar_ndxl_20250630mpcpspeast_d_trt_xx_fp_xx_xxx.tif",
        'vol_path': f"{base_path}/my4rspeast/data/derived/20250630/ch2_sar_ndxl_20250630my4rspeast_d_vol_xx_fp_xx_xxx.tif",
        'odd_path': f"{base_path}/my4rspeast/data/derived/20250630/ch2_sar_ndxl_20250630my4rspeast_d_odd_xx_fp_xx_xxx.tif",
        'evn_path': f"{base_path}/my4rspeast/data/derived/20250630/ch2_sar_ndxl_20250630my4rspeast_d_evn_xx_fp_xx_xxx.tif",
        'hlx_path': f"{base_path}/my4rspeast/data/derived/20250630/ch2_sar_ndxl_20250630my4rspeast_d_hlx_xx_fp_xx_xxx.tif",
        # Faustini AOI: use None to process full data extent (no zero-padding).
        # The AOI intersection logic in Agent 1 will clip to actual data bounds.
        # If you want to clip to Faustini specifically, compute UPS coords for
        # 87.3°S 77°E ± 25km and set them here.
        'faustini_aoi': None,
        'target_crs': 'EPSG:104903',
        'output_dir': 'data/outputs'
    }

    # Verify files exist before running
    print("Verifying input files...")
    for key, path in config.items():
        if key.endswith('_path') and path:
            if not os.path.exists(path):
                print(f"ERROR: File not found: {path}")
                return 1
            else:
                print(f"[+] Found: {os.path.basename(path)}")

    print("\nStarting preprocessing...")

    try:
        # Execute preprocessing
        processor = PreprocessorPrime(config)
        result = processor.run(state)

        print("\n" + "="*60)
        print("PREPROCESSOR PRIME COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Coregistered stack: {result['coregistered_stack']}")
        print(f"Quality mask: {result['quality_mask']}")
        print(f"Quality report: {result['quality_report']}")
        print(f"Polarization mode: {result['polarization_mode']}")
        print(f"CRS: {result['crs']}")
        print(f"Bands processed: {', '.join(result['bands_processed'])}")

        # Show quality report highlights
        import json
        with open(result['quality_report'], 'r') as f:
            report = json.load(f)

        if 'quality_statistics' in report:
            stats = report['quality_statistics']
            print(f"\nQuality Statistics:")
            print(f"  Total pixels: {stats.get('total_pixels', 'N/A'):,}")
            print(f"  Valid pixels: {stats.get('valid_pixels', 'N/A'):,}")
            print(f"  Valid percentage: {stats.get('valid_percentage', 'N/A')}%")

        print("="*60)
        return 0

    except Exception as e:
        print(f"\nERROR: PREPROCESSOR PRIME FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())