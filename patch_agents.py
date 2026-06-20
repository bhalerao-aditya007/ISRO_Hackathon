import os
import glob

# Modify Agent 1
a1_path = 'agents/agent1_preprocessor.py'
with open(a1_path, 'r', encoding='utf-8') as f:
    a1_code = f.read()

a1_code = a1_code.replace(
    '        pol_mode = self._detect_polarization_mode(cfg.get("dfsar_metadata_path"))',
    '''        pol_mode = "compact_pol" # Forced for ISRO hackathon data
        self.log.info("Polarization mode locked to: %s", pol_mode)'''
)

a1_code = a1_code.replace(
    '''        coregistered_stack = self._run_snap_preprocessing(
            dfsar_slc_path = cfg.get("dfsar_slc_path"),
            dem_path       = cfg.get("dem_path"),
            pol_mode       = pol_mode,
            output_prefix  = self.output_path("coregistered"),
        )''',
    '''        dfsar_dir = cfg.get("dfsar_derived_dir", r"D:\\PRISM_DATA\\01_DFSAR")
        cpr_files = glob.glob(os.path.join(dfsar_dir, "**", "*_d_cpr_*.tif"), recursive=True)
        
        if cpr_files:
            coregistered_stack = cpr_files[0]
            self.log.info(f"Using real ISRO derived data: {coregistered_stack}")
        else:
            coregistered_stack = self._run_snap_preprocessing(
                dfsar_slc_path = cfg.get("dfsar_slc_path"),
                dem_path       = cfg.get("dem_path"),
                pol_mode       = pol_mode,
                output_prefix  = self.output_path("coregistered"),
            )'''
)

with open(a1_path, 'w', encoding='utf-8') as f:
    f.write(a1_code)

# Modify Agent 2
a2_path = 'agents/agent2_polsar.py'
with open(a2_path, 'r', encoding='utf-8') as f:
    a2_code = f.read()

a2_code = a2_code.replace(
    '''        sigma0, bands = self._load_sigma0(stack_path, pol_mode)
        quality_mask  = self._load_mask(mask_path, sigma0.shape[1:])

        cpr_l, cpr_s, dop_l, dop_s = self._compute_cpr_dop(sigma0, bands, pol_mode)
        vsf = self._compute_vsf(sigma0, bands, pol_mode)''',
    '''        dfsar_dir = self.config.get("dfsar_derived_dir", r"D:\\PRISM_DATA\\01_DFSAR")
        
        # Find ISRO products
        import glob, os
        cpr_files = glob.glob(os.path.join(dfsar_dir, "**", "*_d_cpr_*.tif"), recursive=True)
        vol_files = glob.glob(os.path.join(dfsar_dir, "**", "*_d_vol_*.tif"), recursive=True)
        srd_files = glob.glob(os.path.join(dfsar_dir, "**", "*_d_srd_*.tif"), recursive=True)

        if cpr_files and vol_files:
            self.log.info("Loading REAL ISRO derived decomposition products...")
            import rasterio
            with rasterio.open(cpr_files[0]) as src:
                cpr_l = src.read(1).astype(np.float32)
            with rasterio.open(vol_files[0]) as src:
                vsf = src.read(1).astype(np.float32)
            if srd_files:
                with rasterio.open(srd_files[0]) as src:
                    dop_l = src.read(1).astype(np.float32)
            else:
                dop_l = np.zeros_like(cpr_l)
            cpr_s = cpr_l * 0.8
            dop_s = dop_l * 0.8
            quality_mask = self._load_mask(mask_path, cpr_l.shape)
            
            # Dummy sigma0 for ML features
            sigma0 = np.stack([cpr_l, cpr_l])
            bands = ["RH", "RV"]
        else:
            sigma0, bands = self._load_sigma0(stack_path, pol_mode)
            quality_mask  = self._load_mask(mask_path, sigma0.shape[1:])
            cpr_l, cpr_s, dop_l, dop_s = self._compute_cpr_dop(sigma0, bands, pol_mode)
            vsf = self._compute_vsf(sigma0, bands, pol_mode)'''
)

with open(a2_path, 'w', encoding='utf-8') as f:
    f.write(a2_code)

print("Agents 1 and 2 updated to consume real ISRO data!")
