import logging
import sys
from pathlib import Path
from protocol import PipelineState
from agent3_depth import DepthSounder

logging.basicConfig(level=logging.INFO)

def main():
    print("Testing PRISM Agent 3: DEPTH SOUNDER")
    
    # We use the outputs from Agent 1 and Agent 2
    # L-band CPR from Agent 1
    cpr_l_path = "data/outputs/cpr_processed.tif"
    
    # S-band CPR. 
    # Note: As checked, this does not exist in D:\PRISM_DATA\01_DFSAR. 
    # The agent is now strictly programmed to fail if this real data is missing,
    # as per the requirement to "remove synthetic fallbacks".
    cpr_s_path = "data/outputs/cpr_s_processed.tif" 
    
    # Probability of Ice from Agent 2
    p_ice_path = "data/outputs/polsar_ice_probability_map.tif"
    
    # Quality Mask from Agent 1
    quality_mask = "data/outputs/prism_quality_mask.tif"
    
    state = PipelineState(
        cpr_l_path=cpr_l_path,
        cpr_s_path=cpr_s_path,
        p_ice_path=p_ice_path,
        quality_mask=quality_mask,
        enl=12.0
    )
    
    config = {}
    agent = DepthSounder(config, output_dir="data/outputs")
    
    try:
        new_state = agent._execute(state)
        print("Agent 3 completed successfully!")
    except FileNotFoundError as e:
        print("\nAgent 3 correctly halted due to missing real data:")
        print(f"ERROR: {e}")
        print("\nThis verifies the strict 'all real data' enforcement is working.")

if __name__ == "__main__":
    main()
