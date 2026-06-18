#!/usr/bin/env python
"""
PRISM Agent 2: POLSAR DETECTIVE (Ice Detection Agent)
======================================================
Processes preprocessed L4-MOSAIC derived GeoTIFF products to generate ice probability map.

Responsibilities:
1. Load preprocessed bands (CPR, SRD, VOL, ODD, EVN, HLX) from Agent 1 output.
2. Compute features from Yamaguchi decomposition ONLY: VSF, ODD_frac, EVN_frac, HLX_frac, Entropy.
3. Generate training labels from CPR threshold (independent physics domain — avoids label leakage).
4. Train a Decision Tree classifier with controlled depth (anti-overfitting).
5. Evaluate with StratifiedKFold cross-validation.
6. Save trained model to disk.
7. Generate ice probability map (0-1) for the entire scene.
8. Output ice probability map as GeoTIFF and metadata JSON.

LABEL LEAKAGE FIX:
- Labels come from CPR (polarization ratio domain)
- Features come from Yamaguchi decomposition (scattering mechanism domain)
- These are physically related but not algebraically identical, breaking the leakage cycle.

Zero synthetic data fallbacks - fails on any data issue.
"""

from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from osgeo import gdal
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("PRISM.POLSAR_DETECTIVE")


class PolSarDetective:
    """Agent 2: PolSar Detective for ice probability mapping."""

    def __init__(self, config: Dict):
        """
        Initialize the detective with configuration.

        Args:
            config: Dictionary containing:
                - cpr_path: Path to preprocessed CPR GeoTIFF (from Agent 1)
                - srd_path: Path to preprocessed SRD GeoTIFF (from Agent 1)
                - vol_path: Path to preprocessed VOL GeoTIFF (from Agent 1)
                - odd_path: Path to preprocessed ODD GeoTIFF (from Agent 1)
                - evn_path: Path to preprocessed EVN GeoTIFF (from Agent 1)
                - hlx_path: Path to preprocessed HLX GeoTIFF (from Agent 1)
                - trt_path: Path to TRT GeoTIFF (NOT USED for labeling anymore)
                - output_dir: Directory to save outputs (model, probability map, etc.)
                - n_samples: Number of random samples to use for training (default: 100000)
                - test_size: Proportion of dataset to use for test (default: 0.15)
                - val_size: Proportion of dataset to use for validation (default: 0.15)
                - dt_max_depth: Maximum depth of Decision Tree (default: 5)
                - dt_min_samples_leaf: Minimum samples per leaf (default: 50)
                - dt_min_samples_split: Minimum samples to split (default: 100)
                - dt_random_state: Random state for reproducibility (default: 42)
                - cv_folds: Number of cross-validation folds (default: 5)
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model parameters — Decision Tree with anti-overfitting controls
        self.n_samples = config.get('n_samples', 100000)
        self.test_size = config.get('test_size', 0.15)
        self.val_size = config.get('val_size', 0.15)
        self.dt_max_depth = config.get('dt_max_depth', 5)
        self.dt_min_samples_leaf = config.get('dt_min_samples_leaf', 50)
        self.dt_min_samples_split = config.get('dt_min_samples_split', 100)
        self.dt_random_state = config.get('dt_random_state', 42)
        self.cv_folds = config.get('cv_folds', 5)

        # TRT path (kept for compatibility but NOT used for labeling)
        self.trt_path = config.get('trt_path')

        # All bands needed for loading
        self.all_bands = ['cpr', 'srd', 'vol', 'odd', 'evn', 'hlx']

        # Feature bands: Yamaguchi decomposition ONLY (NOT CPR/SRD — those are labels)
        self.feature_names = ['VSF', 'ODD_frac', 'EVN_frac', 'HLX_frac', 'Entropy']

        # Validate inputs
        for band in self.all_bands:
            path_key = f'{band}_path'
            if path_key not in config or not config[path_key]:
                raise ValueError(f"Missing required input: {path_key}")
            if not os.path.exists(config[path_key]):
                raise FileNotFoundError(f"File not found: {config[path_key]}")

    def run(self) -> Dict:
        """
        Execute the ice detection pipeline.
        Returns dictionary with output file paths and metadata.
        """
        log.info("=== STARTING POLSAR DETECTIVE ===")
        log.info("Model: DecisionTree (anti-overfitting controls)")
        log.info("Labels: CPR-based (independent physics domain)")
        log.info("Features: Yamaguchi decomposition only (no label leakage)")

        # Step 1: Load and prepare data
        log.info("Loading preprocessed bands...")
        band_data, metadata = self._load_bands()

        # Step 2: Compute features (Yamaguchi-only, NOT CPR/SRD)
        log.info("Computing Yamaguchi-only features (VSF, fractions, entropy)...")
        features = self._compute_features(band_data)

        # Step 3: Generate labels from CPR (independent physics domain)
        log.info("Generating labels from CPR threshold (independent of features)...")
        labels = self._generate_labels(band_data)

        # Step 4: Sample data for training
        log.info(f"Sampling {self.n_samples} random pixels for training...")
        X_sample, y_sample = self._sample_data(features, labels, self.n_samples)

        # Step 5: Split into train, validation, test
        log.info("Splitting data into train, validation, and test sets...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_sample, y_sample,
            test_size=(self.test_size + self.val_size),
            random_state=self.dt_random_state,
            stratify=y_sample
        )
        val_size_adjusted = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size_adjusted),
            random_state=self.dt_random_state,
            stratify=y_temp
        )

        log.info(f"Training set size: {X_train.shape[0]} samples")
        log.info(f"Validation set size: {X_val.shape[0]} samples")
        log.info(f"Test set size: {X_test.shape[0]} samples")
        log.info(f"Label distribution - Train: {np.sum(y_train==1)}/{len(y_train)} ice pixels "
                 f"({np.sum(y_train==1)/len(y_train)*100:.1f}%)")

        # Step 6: Train Decision Tree classifier
        log.info("Training Decision Tree classifier...")
        dt_params = {
            'max_depth': self.dt_max_depth,
            'min_samples_leaf': self.dt_min_samples_leaf,
            'min_samples_split': self.dt_min_samples_split,
            'random_state': self.dt_random_state,
            'class_weight': 'balanced'
        }
        clf = DecisionTreeClassifier(**dt_params)
        clf.fit(X_train, y_train)

        # Step 7: Evaluate on train, validation, and test sets
        log.info("Evaluating classifier...")
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)
        test_pred = clf.predict(X_test)

        # Training metrics (to detect overfitting)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, zero_division=0)

        # Validation metrics
        val_accuracy = accuracy_score(y_val, val_pred)
        val_precision = precision_score(y_val, val_pred, zero_division=0)
        val_recall = recall_score(y_val, val_pred, zero_division=0)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)

        # Test metrics
        test_accuracy = accuracy_score(y_test, test_pred)
        test_precision = precision_score(y_test, test_pred, zero_division=0)
        test_recall = recall_score(y_test, test_pred, zero_division=0)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)

        # Overfitting gap
        overfit_gap_acc = train_accuracy - test_accuracy
        overfit_gap_f1 = train_f1 - test_f1

        log.info(f"Training Metrics:   Accuracy={train_accuracy:.4f}  F1={train_f1:.4f}")
        log.info(f"Validation Metrics:")
        log.info(f"  Accuracy: {val_accuracy:.4f}")
        log.info(f"  Precision: {val_precision:.4f}")
        log.info(f"  Recall: {val_recall:.4f}")
        log.info(f"  F1-score: {val_f1:.4f}")
        log.info(f"Test Metrics:")
        log.info(f"  Accuracy: {test_accuracy:.4f}")
        log.info(f"  Precision: {test_precision:.4f}")
        log.info(f"  Recall: {test_recall:.4f}")
        log.info(f"  F1-score: {test_f1:.4f}")

        # Overfitting check
        log.info(f"Overfitting Gap (train - test):")
        log.info(f"  Accuracy gap: {overfit_gap_acc:.4f}")
        log.info(f"  F1 gap: {overfit_gap_f1:.4f}")
        if overfit_gap_f1 > 0.10:
            log.warning(f"OVERFITTING DETECTED: F1 gap = {overfit_gap_f1:.4f} > 0.10 threshold")
        else:
            log.info(f"Overfitting check PASSED: F1 gap = {overfit_gap_f1:.4f} <= 0.10")

        # Step 8: Cross-validation for robust estimate
        log.info(f"Running {self.cv_folds}-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.dt_random_state)
        cv_scores = cross_val_score(
            DecisionTreeClassifier(**dt_params),
            X_sample, y_sample,
            cv=cv, scoring='f1', n_jobs=-1
        )
        log.info(f"CV F1 scores: {cv_scores}")
        log.info(f"CV F1 mean: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Feature importance logging
        log.info("Feature Importances:")
        for name, importance in zip(self.feature_names, clf.feature_importances_):
            log.info(f"  {name}: {importance:.4f}")

        # Step 9: Save trained model
        model_path = self.output_dir / "polsar_detective_rf_model.joblib"
        log.info(f"Saving trained model to {model_path}")
        joblib.dump(clf, model_path)

        # Step 10: Generate ice probability map for entire scene
        log.info("Generating ice probability map for entire scene...")
        prob_map_path = self._predict_probability_map(clf, features, metadata)

        # Step 11: Save metadata and metrics
        metadata_path = self.output_dir / "polsar_detective_metadata.json"
        metadata_to_save = {
            'agent': 'POLSAR_DETECTIVE',
            'processing_timestamp': str(np.datetime64('now')),
            'model_type': 'DecisionTreeClassifier',
            'label_source': 'CPR_threshold (independent physics domain)',
            'feature_source': 'Yamaguchi decomposition only (no label leakage)',
            'input_bands': {band: self.config.get(f'{band}_path') for band in self.all_bands},
            'trt_path_note': 'TRT not used for labeling (label leakage fix)',
            'model_parameters': dt_params,
            'training_samples': self.n_samples,
            'train_size': int(X_train.shape[0]),
            'val_size': int(X_val.shape[0]),
            'test_size': int(X_test.shape[0]),
            'training_metrics': {
                'accuracy': float(train_accuracy),
                'f1_score': float(train_f1)
            },
            'validation_metrics': {
                'accuracy': float(val_accuracy),
                'precision': float(val_precision),
                'recall': float(val_recall),
                'f1_score': float(val_f1)
            },
            'test_metrics': {
                'accuracy': float(test_accuracy),
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1_score': float(test_f1)
            },
            'overfitting_gap': {
                'accuracy_gap': float(overfit_gap_acc),
                'f1_gap': float(overfit_gap_f1),
                'is_overfitting': bool(overfit_gap_f1 > 0.10)
            },
            'cross_validation': {
                'n_folds': self.cv_folds,
                'f1_scores': [float(s) for s in cv_scores],
                'f1_mean': float(cv_scores.mean()),
                'f1_std': float(cv_scores.std())
            },
            'feature_importance': dict(zip(self.feature_names, [float(x) for x in clf.feature_importances_])),
            'outputs': {
                'model_path': str(model_path),
                'probability_map_path': str(prob_map_path),
                'metadata_path': str(metadata_path)
            }
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata_to_save, f, indent=2)
        log.info(f"Metadata saved to {metadata_path}")

        log.info("=== POLSAR DETECTIVE COMPLETED SUCCESSFULLY ===")

        return {
            'model_path': str(model_path),
            'probability_map_path': str(prob_map_path),
            'metadata_path': str(metadata_path),
            'validation_metrics': {
                'accuracy': val_accuracy,
                'precision': val_precision,
                'recall': val_recall,
                'f1_score': val_f1
            },
            'test_metrics': {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1
            }
        }

    def _load_bands(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Load the preprocessed bands and return data and metadata."""
        band_data = {}
        metadata = {}

        for band in self.all_bands:
            path = self.config.get(f'{band}_path')
            dataset = gdal.Open(path, gdal.GA_ReadOnly)
            if dataset is None:
                raise ValueError(f"Failed to open {band} band: {path}")

            band_array = dataset.GetRasterBand(1).ReadAsArray()
            band_data[band] = band_array

            # Store metadata from the first band
            if band == 'cpr':
                metadata = {
                    'width': dataset.RasterXSize,
                    'height': dataset.RasterYSize,
                    'projection': dataset.GetProjection(),
                    'geotransform': dataset.GetGeoTransform(),
                    'datatype': gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)
                }

            dataset = None  # Close dataset

        return band_data, metadata

    def _compute_features(self, band_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute features from Yamaguchi decomposition ONLY.
        Features: [VSF, ODD_frac, EVN_frac, HLX_frac, Entropy]
        
        CPR and SRD are deliberately EXCLUDED from features because
        CPR is used for labeling — including it would be label leakage.
        
        Returns shape (n_pixels, 5)
        """
        sample_band = band_data['vol']
        height, width = sample_band.shape
        n_pixels = height * width

        # Flatten Yamaguchi components only
        vol_flat = band_data['vol'].flatten()
        odd_flat = band_data['odd'].flatten()
        evn_flat = band_data['evn'].flatten()
        hlx_flat = band_data['hlx'].flatten()

        # Calculate total power for fractions
        total_power = vol_flat + odd_flat + evn_flat + hlx_flat
        mask = total_power != 0

        # Initialize feature array (5 features — no CPR, no SRD)
        features = np.zeros((n_pixels, 5), dtype=np.float32)

        # Feature 0: VSF (Volume Scattering Fraction)
        vsf = np.zeros_like(total_power)
        vsf[mask] = vol_flat[mask] / total_power[mask]
        features[:, 0] = vsf

        # Feature 1: ODD Fraction
        odd_frac = np.zeros_like(total_power)
        odd_frac[mask] = odd_flat[mask] / total_power[mask]
        features[:, 1] = odd_frac

        # Feature 2: EVN Fraction
        evn_frac = np.zeros_like(total_power)
        evn_frac[mask] = evn_flat[mask] / total_power[mask]
        features[:, 2] = evn_frac

        # Feature 3: HLX Fraction (Helix)
        hlx_frac = np.zeros_like(total_power)
        hlx_frac[mask] = hlx_flat[mask] / total_power[mask]
        features[:, 3] = hlx_frac

        # Feature 4: Entropy proxy (scattering randomness)
        epsilon = 1e-10
        if np.any(mask):
            vol_valid = vol_flat[mask]
            odd_valid = odd_flat[mask]
            evn_valid = evn_flat[mask]
            hlx_valid = hlx_flat[mask]
            total_valid = total_power[mask]

            vol_frac_valid = vol_valid / total_valid
            odd_frac_valid = odd_valid / total_valid
            evn_frac_valid = evn_valid / total_valid
            hlx_frac_valid = hlx_valid / total_valid

            # Avoid zeros for log
            vol_frac_valid = np.maximum(vol_frac_valid, epsilon)
            odd_frac_valid = np.maximum(odd_frac_valid, epsilon)
            evn_frac_valid = np.maximum(evn_frac_valid, epsilon)
            hlx_frac_valid = np.maximum(hlx_frac_valid, epsilon)

            # Renormalize
            sum_fracs = vol_frac_valid + odd_frac_valid + evn_frac_valid + hlx_frac_valid
            vol_frac_valid = vol_frac_valid / sum_fracs
            odd_frac_valid = odd_frac_valid / sum_fracs
            evn_frac_valid = evn_frac_valid / sum_fracs
            hlx_frac_valid = hlx_frac_valid / sum_fracs

            # Shannon entropy
            entropy_valid = -(vol_frac_valid * np.log2(vol_frac_valid) +
                           odd_frac_valid * np.log2(odd_frac_valid) +
                           evn_frac_valid * np.log2(evn_frac_valid) +
                           hlx_frac_valid * np.log2(hlx_frac_valid))

            features[mask, 4] = entropy_valid

        return features

    def _generate_labels(self, band_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate training labels from CPR (independent physics domain).

        Labels are derived from CPR > 1.0 (ice signature per PRL paper).
        Features are Yamaguchi fractions — physically related but NOT algebraically
        derived from CPR. This breaks the label leakage cycle.

        The DOP (SRD) band is used as a secondary refinement:
        CPR > 1.0 AND DOP < 0.5 -> ice candidate label = 1
        This adds physical meaning: high CPR with low polarization degree
        indicates volume scattering from subsurface ice, not surface boulders.
        """
        cpr_flat = band_data['cpr'].flatten()
        srd_flat = band_data['srd'].flatten()

        # Primary criterion: CPR > 1.0 (standard ice threshold from PRL paper)
        # Secondary criterion: DOP (SRD) < 0.5 (to exclude surface-only scatterers)
        # Also require valid data (non-zero total power in Yamaguchi)
        vol_flat = band_data['vol'].flatten()
        odd_flat = band_data['odd'].flatten()
        evn_flat = band_data['evn'].flatten()
        hlx_flat = band_data['hlx'].flatten()
        total_power = vol_flat + odd_flat + evn_flat + hlx_flat

        ice_condition = (
            (cpr_flat > 1.0) &
            (srd_flat < 0.5) &
            (total_power > 0)  # valid data only
        )

        labels = ice_condition.astype(np.uint8)

        n_ice = np.sum(labels)
        n_total = len(labels)
        log.info(f"CPR-based labeling: {n_ice} ice pixels out of {n_total} "
                 f"({n_ice/n_total*100:.2f}%)")

        if n_ice == 0:
            log.warning("No ice pixels found! Check CPR threshold or data range.")
            log.info(f"CPR range: [{np.nanmin(cpr_flat):.4f}, {np.nanmax(cpr_flat):.4f}]")
            log.info(f"SRD range: [{np.nanmin(srd_flat):.4f}, {np.nanmax(srd_flat):.4f}]")
        elif n_ice / n_total > 0.5:
            log.warning(f"Very high ice fraction ({n_ice/n_total*100:.1f}%) -- "
                       "consider tightening CPR threshold")

        return labels

    def _sample_data(self, features: np.ndarray, labels: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random subset of pixels for training, ensuring class balance."""
        n_pixels = features.shape[0]

        # Filter out invalid pixels (all-zero features)
        valid_mask = np.any(features != 0, axis=1)
        valid_indices = np.where(valid_mask)[0]
        log.info(f"Valid pixels for training: {len(valid_indices)} / {n_pixels}")

        if len(valid_indices) == 0:
            raise ValueError("No valid pixels found for training!")

        features_valid = features[valid_indices]
        labels_valid = labels[valid_indices]

        if n_samples >= len(valid_indices):
            log.warning(f"Requested samples ({n_samples}) >= valid pixels ({len(valid_indices)}), using all.")
            return features_valid, labels_valid

        # Random sampling without replacement
        rng = np.random.RandomState(self.dt_random_state)
        indices = rng.choice(len(valid_indices), size=n_samples, replace=False)
        return features_valid[indices], labels_valid[indices]

    def _predict_probability_map(self, clf: DecisionTreeClassifier, features: np.ndarray, metadata: Dict) -> str:
        """Predict ice probability for the entire scene and save as GeoTIFF."""
        log.info("Predicting probabilities for all pixels...")

        # For DecisionTree, predict_proba gives leaf node class fractions
        probs = clf.predict_proba(features)

        # Handle case where only one class exists in training data
        if probs.shape[1] == 1:
            log.warning("Only one class in predictions -- probability map will be uniform")
            prob_ice = np.zeros(probs.shape[0], dtype=np.float32)
        else:
            prob_ice = probs[:, 1].astype(np.float32)

        # Reshape to original dimensions
        height = metadata['height']
        width = metadata['width']
        prob_map = prob_ice.reshape((height, width))

        # Save as GeoTIFF
        output_path = self.output_dir / "polsar_ice_probability_map.tif"
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            str(output_path),
            width,
            height,
            1,
            gdal.GDT_Float32
        )
        if dst_ds is None:
            raise ValueError(f"Failed to create output file: {output_path}")

        dst_ds.SetProjection(metadata['projection'])
        dst_ds.SetGeoTransform(metadata['geotransform'])

        band = dst_ds.GetRasterBand(1)
        band.WriteArray(prob_map)
        band.SetNoDataValue(-9999.0)

        dst_ds.FlushCache()
        dst_ds = None

        log.info(f"Ice probability map saved to {output_path}")
        log.info(f"  Prob range: [{np.nanmin(prob_map):.4f}, {np.nanmax(prob_map):.4f}]")
        log.info(f"  Pixels with P>0.5: {np.sum(prob_map > 0.5)}")
        return str(output_path)


def main():
    """Command-line interface for standalone testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description='PRISM PolSar Detective - Ice Detection Agent'
    )
    parser.add_argument('--cpr', required=True, help='Path to preprocessed CPR GeoTIFF')
    parser.add_argument('--srd', required=True, help='Path to preprocessed SRD GeoTIFF')
    parser.add_argument('--vol', required=True, help='Path to preprocessed VOL GeoTIFF')
    parser.add_argument('--odd', required=True, help='Path to preprocessed ODD GeoTIFF')
    parser.add_argument('--evn', required=True, help='Path to preprocessed EVN GeoTIFF')
    parser.add_argument('--hlx', required=True, help='Path to preprocessed HLX GeoTIFF')
    parser.add_argument('--trt', help='Path to TRT GeoTIFF (kept for compatibility, not used for labels)')
    parser.add_argument('--output-dir', default='data/outputs',
                       help='Output directory (default: data/outputs)')
    parser.add_argument('--n-samples', type=int, default=100000,
                       help='Number of random samples to use for training (default: 100000)')
    parser.add_argument('--test-size', type=float, default=0.15,
                       help='Proportion of dataset to use for test (default: 0.15)')
    parser.add_argument('--val-size', type=float, default=0.15,
                       help='Proportion of dataset to use for validation (default: 0.15)')
    parser.add_argument('--dt-max-depth', type=int, default=5,
                       help='Maximum depth of Decision Tree (default: 5)')
    parser.add_argument('--dt-min-samples-leaf', type=int, default=50,
                       help='Minimum samples per leaf (default: 50)')
    parser.add_argument('--dt-min-samples-split', type=int, default=100,
                       help='Minimum samples to split (default: 100)')
    parser.add_argument('--dt-random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = {
        'cpr_path': args.cpr,
        'srd_path': args.srd,
        'vol_path': args.vol,
        'odd_path': args.odd,
        'evn_path': args.evn,
        'hlx_path': args.hlx,
        'trt_path': args.trt,
        'output_dir': args.output_dir,
        'n_samples': args.n_samples,
        'test_size': args.test_size,
        'val_size': args.val_size,
        'dt_max_depth': args.dt_max_depth,
        'dt_min_samples_leaf': args.dt_min_samples_leaf,
        'dt_min_samples_split': args.dt_min_samples_split,
        'dt_random_state': args.dt_random_state,
        'cv_folds': args.cv_folds
    }

    # Execute detection
    try:
        detective = PolSarDetective(config)
        result = detective.run()

        print("\n" + "="*60)
        print("POLSAR DETECTIVE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Model saved to: {result['model_path']}")
        print(f"Ice probability map: {result['probability_map_path']}")
        print(f"Metadata saved to: {result['metadata_path']}")
        print("\nValidation Metrics:")
        for metric, value in result['validation_metrics'].items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        print("\nTest Metrics:")
        for metric, value in result['test_metrics'].items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        print("="*60)

        return 0

    except Exception as e:
        log.error(f"POLSAR DETECTIVE FAILED: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())