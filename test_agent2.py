import os
from pathlib import Path
from agents.agent2_polsar_detective import PolSarDetective

def main():
    print("Testing Agent 2: PolSar Detective")
    
    # Path to outputs from Agent 1
    base_dir = Path('data/outputs')
    
    config = {
        'cpr_path': str(base_dir / 'cpr_processed.tif'),
        'srd_path': str(base_dir / 'srd_processed.tif'),
        'vol_path': str(base_dir / 'vol_processed.tif'),
        'odd_path': str(base_dir / 'odd_processed.tif'),
        'evn_path': str(base_dir / 'evn_processed.tif'),
        'hlx_path': str(base_dir / 'hlx_processed.tif'),
        'trt_path': str(base_dir / 'trt_processed.tif'),
        'output_dir': str(base_dir),
        'n_samples': 50000,
        'cv_folds': 3  # Faster for testing
    }
    
    try:
        detective = PolSarDetective(config)
        result = detective.run()
        print(f"Agent 2 Test SUCCESS. Model saved to: {result['model_path']}")
    except Exception as e:
        print(f"Agent 2 Test FAILED: {e}")

if __name__ == "__main__":
    main()
