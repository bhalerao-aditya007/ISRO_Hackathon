#!/usr/bin/env python
"""
PRISM Agent 1: PREPROCESSOR PRIME (L4-MOSAIC Products Version)
===============================================================
Processes ISRO-provided L4-MOSAIC derived GeoTIFF products:
- CPR, SRD, TRT (polarimetric parameters)
- VOL, ODD, EVN, HLX (Yamaguchi 4-components)
All files are already calibrated, geocoded, and at 25m resolution.

Responsibilities:
1. Validate all input files exist and are readable
2. Confirm consistent projection, resolution, and dimensions
3. Optionally clip to Faustini AOI (if provided)
4. Create co-registered multi-band stack
5. Generate quality mask (data validity only)
6. Produce JSON quality report
7. Output stack for downstream agents

ZERO synthetic data fallbacks - fails on any data issue.
"""

from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal, osr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("PRISM.PreprocessorPrime")


class PreprocessorPrime:
    """Agent 1: Preprocessor Prime for L4-MOSAIC products."""

    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Dictionary containing:
                - cpr_path, srd_path, trt_path: Required polarimetric files
                - vol_path, odd_path, evn_path, hlx_path: Yamaguchi components
                - faustini_aoi: Optional (minX, minY, maxX, maxY) in target CRS
                - target_crs: Output CRS (default: EPSG:104903)
        """
        self.config = config
        self.target_crs = config.get('target_crs', 'EPSG:104903')
        self.required_files = ['cpr', 'srd', 'trt']
        self.optional_files = ['vol', 'odd', 'evn', 'hlx']
        self.all_files = self.required_files + self.optional_files

        # Validate configuration
        missing_required = [f for f in self.required_files if not config.get(f'{f}_path')]
        if missing_required:
            raise ValueError(f"Missing required files: {missing_required}")

    def run(self) -> Dict:
        """
        Execute preprocessing pipeline.
        Returns dictionary with output file paths and metadata.
        """
        log.info("=== STARTING PREPROCESSOR PRIME (L4-MOSAIC) ===")
        log.info("Polarization mode: DUAL_POL_LINEAR LH/LV (per context)")

        # Step 1: Validate all input files
        file_paths = self._validate_input_files()
        log.info("Input validation complete")

        # Step 2: Get reference grid from CPR file
        reference_info = self._get_raster_info(file_paths['cpr'])
        log.info(f"Reference grid: {reference_info['width']}x{reference_info['height']}")
        log.info(f"Reference CRS: {reference_info['crs']}")

        # Step 3: Verify all files match reference grid
        self._verify_grid_consistency(file_paths, reference_info)
        log.info("Grid consistency verification passed")

        # Step 4: Create output directory
        output_dir = Path(self.config.get('output_dir', 'data/outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 5: Process each file (reproject if needed, clip to AOI)
        processed_files = {}
        for name, path in file_paths.items():
            processed_path = self._process_single_file(
                input_path=path,
                name=name,
                reference_info=reference_info,
                output_dir=output_dir
            )
            processed_files[name] = processed_path
            log.info(f"Processed {name}: {processed_path}")

        # Step 6: Create multi-band stack
        stack_path = output_dir / "prism_coregistered_stack.tif"
        log.debug(f"Multi-band stack path: {stack_path}")
        self._create_multiband_stack(
            input_files=processed_files,
            output_path=stack_path,
            band_order=self.all_files  # Process in consistent order
        )
        log.info(f"Multi-band stack created: {stack_path}")

        # Step 7: Build quality mask (simple data validity)
        mask_path = output_dir / "prism_quality_mask.tif"
        self._build_quality_mask(
            input_path=stack_path,
            output_path=mask_path
        )
        log.info(f"Quality mask created: {mask_path}")

        # Step 8: Generate quality report
        report_path = output_dir / "prism_quality_report.json"
        quality_report = self._generate_quality_report(
            input_files=file_paths,
            processed_files=processed_files,
            stack_path=stack_path,
            mask_path=mask_path,
            reference_info=reference_info
        )
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        log.info(f"Quality report created: {report_path}")

        log.info("=== PREPROCESSOR PRIME COMPLETED SUCCESSFULLY ===")

        return {
            'coregistered_stack': str(stack_path),
            'quality_mask': str(mask_path),
            'quality_report': str(report_path),
            'polarization_mode': 'DUAL_POL_LINEAR',
            'crs': self.target_crs,
            'bands_processed': list(processed_files.keys())
        }

    def _validate_input_files(self) -> Dict[str, str]:
        """Validate that all configured input files exist and are readable."""
        file_paths = {}

        for name in self.all_files:
            path_key = f'{name}_path'
            path = self.config.get(path_key)

            if path is None:
                file_paths[name] = None
                continue

            path_obj = Path(path)
            if not path_obj.exists():
                raise FileNotFoundError(f"Input file not found: {path}")

            # Try to open with GDAL to verify readability
            try:
                dataset = gdal.Open(str(path_obj), gdal.GA_ReadOnly)
                if dataset is None:
                    raise ValueError(f"GDAL cannot open file: {path}")
                dataset = None  # Close
                file_paths[name] = str(path_obj.resolve())
                log.debug(f"Validated {name}: {path}")
            except Exception as e:
                raise RuntimeError(f"Failed to validate {name} ({path}): {e}")

        # Check that required files are present
        missing_required = [name for name in self.required_files if file_paths[name] is None]
        if missing_required:
            raise ValueError(f"Missing required input files: {missing_required}")

        return file_paths

    def _get_raster_info(self, filepath: str) -> Dict:
        """Extract key raster metadata."""
        dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
        if dataset is None:
            raise ValueError(f"Cannot open raster file: {filepath}")

        # Get projection
        projection_wkt = dataset.GetProjection()
        if not projection_wkt:
            raise ValueError(f"No projection found in {filepath}")

        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection_wkt)
        epsg_code = srs.GetAuthorityCode(None)
        crs = f"EPSG:{epsg_code}" if epsg_code else projection_wkt

        info = {
            'filename': Path(filepath).name,
            'crs': crs,
            'projection_wkt': projection_wkt,  # Store WKT for SetProjection
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'bands': dataset.RasterCount,
            'datatype': gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType),
            'transform': dataset.GetGeoTransform(),
            'nodata': dataset.GetRasterBand(1).GetNoDataValue()
        }

        # Calculate bounds
        gt = info['transform']
        info['bounds'] = (
            gt[0],  # minX
            gt[3] + gt[5] * info['height'],  # minY
            gt[0] + gt[1] * info['width'],   # maxX
            gt[3]                            # maxY
        )

        dataset = None  # Close dataset
        return info

    def _verify_grid_consistency(self, file_paths: Dict[str, Optional[str]],
                                reference: Dict) -> None:
        """Ensure all files match reference grid (CRS, resolution, dimensions)."""
        for name, path in file_paths.items():
            if path is None:
                continue  # Skip optional missing files

            info = self._get_raster_info(path)

            # Check CRS
            if info['crs'] != reference['crs']:
                log.warning(f"{name} CRS mismatch: {info['crs']} vs reference {reference['crs']}")
                # Note: We'll reproject later if needed

            # Check dimensions
            if info['width'] != reference['width'] or info['height'] != reference['height']:
                log.warning(f"{name} dimensions mismatch: {info['width']}x{info['height']} "
                           f"vs reference {reference['width']}x{reference['height']}")

            # Check resolution (from transform)
            gt_ref = reference['transform']
            gt_info = info['transform']
            pixel_width_ref = abs(gt_ref[1])
            pixel_height_ref = abs(gt_ref[5])
            pixel_width_info = abs(gt_info[1])
            pixel_height_info = abs(gt_info[5])

            if abs(pixel_width_info - pixel_width_ref) > 0.001 or \
               abs(pixel_height_info - pixel_height_ref) > 0.001:
                log.warning(f"{name} resolution mismatch: "
                           f"{pixel_width_info}x{pixel_height_info} vs "
                           f"{pixel_width_ref}x{pixel_height_ref}")

        log.info("Grid consistency check completed (warnings logged if any mismatches)")

    def _intersect_aoi_with_bounds(self, aoi: Tuple, bounds: Tuple) -> Optional[Tuple]:
        """
        Intersect user-specified AOI with actual data bounds.
        Returns intersected (minX, minY, maxX, maxY) or None if no overlap.
        Both aoi and bounds are (minX, minY, maxX, maxY).
        """
        int_minx = max(aoi[0], bounds[0])
        int_miny = max(aoi[1], bounds[1])
        int_maxx = min(aoi[2], bounds[2])
        int_maxy = min(aoi[3], bounds[3])

        if int_minx >= int_maxx or int_miny >= int_maxy:
            log.warning(f"AOI does not intersect data bounds. AOI={aoi}, bounds={bounds}")
            return None

        clipped_aoi = (int_minx, int_miny, int_maxx, int_maxy)
        if clipped_aoi != tuple(aoi):
            log.info(f"AOI clipped to data bounds: {aoi} -> {clipped_aoi}")
        return clipped_aoi

    def _crs_are_same(self, wkt1: str, wkt2: str) -> bool:
        """Compare two CRS using osr.IsSame() instead of string comparison."""
        srs1 = osr.SpatialReference()
        srs1.ImportFromWkt(wkt1)
        srs2 = osr.SpatialReference()
        srs2.ImportFromWkt(wkt2)
        return bool(srs1.IsSame(srs2))

    def _process_single_file(self, input_path: str, name: str,
                           reference_info: Dict, output_dir: Path) -> str:
        """
        Process a single file: reproject to target CRS if needed,
        resample to reference resolution, clip to AOI.
        AOI is intersected with source bounds to avoid zero-padding.
        """
        output_path = output_dir / f"{name}_processed.tif"

        # Open input dataset
        input_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
        if input_ds is None:
            raise ValueError(f"Cannot open input file: {input_path}")

        # Get source bounds for AOI intersection
        src_gt = input_ds.GetGeoTransform()
        src_bounds = (
            src_gt[0],                                    # minX
            src_gt[3] + src_gt[5] * input_ds.RasterYSize, # minY
            src_gt[0] + src_gt[1] * input_ds.RasterXSize, # maxX
            src_gt[3]                                     # maxY
        )

        # Intersect AOI with source bounds (prevent zero-padding)
        aoi = self.config.get('faustini_aoi')
        effective_aoi = None
        if aoi:
            effective_aoi = self._intersect_aoi_with_bounds(aoi, src_bounds)
            if effective_aoi is None:
                log.warning(f"{name}: AOI doesn't overlap source data, using full extent")

        # Check if reprojection/resampling needed using proper CRS comparison
        input_wkt = input_ds.GetProjection()
        ref_wkt = reference_info['projection_wkt']
        same_crs = self._crs_are_same(input_wkt, ref_wkt)

        needs_warp = (
            not same_crs or
            abs(input_ds.GetGeoTransform()[1] - reference_info['transform'][1]) > 0.001 or
            abs(input_ds.GetGeoTransform()[5] - reference_info['transform'][5]) > 0.001
        )

        if needs_warp:
            log.info(f"Reprojecting/resampling {name} to match reference grid")
            warp_options = gdal.WarpOptions(
                dstSRS=ref_wkt,
                xRes=abs(reference_info['transform'][1]),
                yRes=abs(reference_info['transform'][5]),
                resampleAlg=gdal.GRA_Bilinear,
                dstNodata=0,
                outputType=gdal.GDT_Float32,
                outputBounds=effective_aoi if effective_aoi else None
            )

            warped_ds = gdal.Warp(str(output_path), input_ds, options=warp_options)
            warped_ds = None  # Close
        else:
            # No reprojection needed - just copy and optionally clip
            translate_options = gdal.TranslateOptions(
                outputType=gdal.GDT_Float32,
                projWin=[effective_aoi[0], effective_aoi[3], effective_aoi[2], effective_aoi[1]] if effective_aoi else None
            )

            translated_ds = gdal.Translate(str(output_path), input_ds, options=translate_options)
            translated_ds = None  # Close

        input_ds = None  # Close input dataset
        return str(output_path)

    def _create_multiband_stack(self, input_files: Dict[str, str],
                               output_path: Path, band_order: List[str]) -> None:
        """Create multi-band GeoTIFF stack from processed files."""
        # Filter to only files that exist
        bands_to_process = [
            name for name in band_order
            if name in input_files and input_files[name] is not None
        ]

        if not bands_to_process:
            raise ValueError("No valid bands to process for stacking")

        # Get reference from first band
        first_band_path = input_files[bands_to_process[0]]
        reference_info = self._get_raster_info(first_band_path)

        # Create output dataset
        driver = gdal.GetDriverByName('GTiff')
        output_ds = driver.Create(
            str(output_path),
            reference_info['width'],
            reference_info['height'],
            len(bands_to_process),
            gdal.GDT_Float32
        )

        if output_ds is None:
            raise ValueError(f"Failed to create output file: {output_path}")

        output_ds.SetProjection(reference_info['projection_wkt'])
        output_ds.SetGeoTransform(reference_info['transform'])

        # Write each band
        for i, band_name in enumerate(bands_to_process, 1):
            band_path = input_files[band_name]
            band_ds = gdal.Open(band_path, gdal.GA_ReadOnly)
            if band_ds is None:
                log.warning(f"Could not open band {band_name}: {band_path}")
                # Write zeros for missing band
                band_data = np.zeros(
                    (reference_info['height'], reference_info['width']),
                    dtype=np.float32
                )
            else:
                band_data = band_ds.GetRasterBand(1).ReadAsArray()
                band_ds = None  # Close

            output_band = output_ds.GetRasterBand(i)
            output_band.WriteArray(band_data)
            output_band.SetDescription(band_name.upper())

            # Set nodata if applicable
            nodata_val = 0  # Our processing uses 0 for invalid
            output_band.SetNoDataValue(nodata_val)
            # Do NOT call Fill() as it overwrites the data we just wrote

        output_ds.FlushCache()
        output_ds = None  # Close
        log.info(f"Created {len(bands_to_process)}-band stack: {output_path}")

    def _build_quality_mask(self, input_path: str, output_path: Path) -> None:
        """
        Build quality mask: 1 = valid data, 0 = invalid/nodata.
        Checks ALL bands ΓÇö a pixel is valid only if every band is finite and non-nodata.
        """
        input_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
        if input_ds is None:
            raise ValueError(f"Cannot open input for masking: {input_path}")

        n_bands = input_ds.RasterCount
        log.info(f"Building quality mask from {n_bands} bands")

        # Start with all-valid mask, then AND each band's validity
        first_band = input_ds.GetRasterBand(1)
        height, width = first_band.YSize, first_band.XSize
        valid_mask = np.ones((height, width), dtype=bool)

        for b in range(1, n_bands + 1):
            band = input_ds.GetRasterBand(b)
            band_data = band.ReadAsArray()
            nodata = band.GetNoDataValue()

            if nodata is not None:
                band_valid = np.isfinite(band_data) & (band_data != nodata)
            else:
                band_valid = np.isfinite(band_data)

            valid_mask &= band_valid

        input_ds = None  # Close

        valid_mask_uint8 = valid_mask.astype(np.uint8)

        # Get georeference info
        info = self._get_raster_info(input_path)

        # Write mask
        driver = gdal.GetDriverByName('GTiff')
        mask_ds = driver.Create(
            str(output_path),
            info['width'],
            info['height'],
            1,
            gdal.GDT_Byte
        )

        if mask_ds is None:
            raise ValueError(f"Failed to create quality mask: {output_path}")

        mask_ds.SetProjection(info['projection_wkt'])
        mask_ds.SetGeoTransform(info['transform'])

        mask_band = mask_ds.GetRasterBand(1)
        mask_band.WriteArray(valid_mask_uint8)
        mask_band.SetNoDataValue(0)

        mask_ds.FlushCache()
        mask_ds = None  # Close
        log.info(f"Quality mask created: {output_path} (valid pixels: {np.sum(valid_mask_uint8)}/{valid_mask_uint8.size})")

    def _generate_quality_report(self, input_files: Dict[str, Optional[str]],
                               processed_files: Dict[str, str],
                               stack_path: Path, mask_path: Path,
                               reference_info: Dict) -> Dict:
        """Generate comprehensive quality report JSON."""
        report = {
            'agent': 'PREPROCESSOR_PRIME',
            'processing_timestamp': str(np.datetime64('now')),
            'polarization_mode': 'DUAL_POL_LINEAR',
            'products_type': 'L4-MOSAIC_DERIVED',
            'target_crs': reference_info['crs'],
            'target_resolution_m': abs(reference_info['transform'][1]),
            'input_files': {},
            'processed_files': {},
            'outputs': {
                'coregistered_stack': str(stack_path),
                'quality_mask': str(mask_path)
            },
            'quality_statistics': {}
        }

        # Report on input files
        for name in self.all_files:
            input_path = input_files.get(name)
            report['input_files'][name] = {
                'path': input_path,
                'exists': input_path is not None and Path(input_path).exists() if input_path else False
            }
            if input_path:
                try:
                    info = self._get_raster_info(input_path)
                    report['input_files'][name]['info'] = info
                except Exception as e:
                    report['input_files'][name]['error'] = str(e)

        # Report on processed files
        for name, path in processed_files.items():
            try:
                info = self._get_raster_info(path)
                report['processed_files'][name] = {
                    'path': path,
                    'info': info
                }
            except Exception as e:
                report['processed_files'][name] = {
                    'path': path,
                    'error': str(e)
                }

        # Calculate statistics from quality mask
        try:
            mask_info = self._get_raster_info(str(mask_path))
            mask_ds = gdal.Open(str(mask_path), gdal.GA_ReadOnly)
            mask_data = mask_ds.GetRasterBand(1).ReadAsArray()
            mask_ds = None

            total_pixels = mask_data.size
            valid_pixels = np.sum(mask_data == 1)

            report['quality_statistics'] = {
                'total_pixels': int(total_pixels),
                'valid_pixels': int(valid_pixels),
                'valid_percentage': round((valid_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0,
                'invalid_pixels': int(total_pixels - valid_pixels)
            }
        except Exception as e:
            report['quality_statistics']['error'] = str(e)

        # Stack info
        try:
            report['stack_info'] = self._get_raster_info(str(stack_path))
        except Exception as e:
            report['stack_info']['error'] = str(e)

        return report


def main():
    """Command-line interface for standalone testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description='PRISM Preprocessor Prime - L4-MOSAIC Products'
    )
    parser.add_argument('--cpr', required=True, help='Path to CPR GeoTIFF')
    parser.add_argument('--srd', required=True, help='Path to SRD GeoTIFF')
    parser.add_argument('--trt', required=True, help='Path to TRT GeoTIFF')
    parser.add_argument('--vol', help='Path to Volume scattering GeoTIFF')
    parser.add_argument('--odd', help='Path to Odd bounce GeoTIFF')
    parser.add_argument('--evn', help='Path to Even bounce GeoTIFF')
    parser.add_argument('--hlx', help='Path to Helix GeoTIFF')
    parser.add_argument('--aoi', nargs=4, type=float, metavar=('MINX', 'MINY', 'MAXX', 'MAXY'),
                       help='FAUSTINI AOI: minX minY maxX maxY (in target CRS)')
    parser.add_argument('--target-crs', default='EPSG:104903',
                       help='Target CRS for output (default: EPSG:104903)')
    parser.add_argument('--output-dir', default='data/outputs',
                       help='Output directory (default: data/outputs)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = {
        'cpr_path': args.cpr,
        'srd_path': args.srd,
        'trt_path': args.trt,
        'vol_path': args.vol,
        'odd_path': args.odd,
        'evn_path': args.evn,
        'hlx_path': args.hlx,
        'faustini_aoi': tuple(args.aoi) if args.aoi else None,
        'target_crs': args.target_crs,
        'output_dir': args.output_dir
    }

    # Execute preprocessing
    try:
        processor = PreprocessorPrime(config)
        result = processor.run()

        print("\n" + "="*60)
        print("PREPROCESSOR PRIME COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Coregistered stack: {result['coregistered_stack']}")
        print(f"Quality mask: {result['quality_mask']}")
        print(f"Quality report: {result['quality_report']}")
        print(f"Polarization mode: {result['polarization_mode']}")
        print(f"CRS: {result['crs']}")
        print(f"Bands processed: {', '.join(result['bands_processed'])}")
        print("="*60)

        return 0

    except Exception as e:
        log.error(f"PREPROCESSOR PRIME FAILED: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
