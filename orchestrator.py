"""
PRISM — Pipeline Orchestrator
===============================
LangGraph DAG executing all 8 agents in the correct dependency order.
Falls back to a plain sequential runner if LangGraph is not installed.

DAG topology (from PRISM spec):
  START
    → Agent 1 (PREPROCESSOR_PRIME)
    → [Agent 2, Agent 4, Agent 6 pass1]    (parallel)
    → Agent 3  (waits for Agent 2)
    → Agent 5  (waits for Agent 3 + Agent 2)
    → Agent 7  (waits for Agent 3, 4, 5, 6)
    → Agent 6 pass2 (waits for Agent 7)
    → Agent 8  (waits for Agent 6 pass2 + Agent 7)
    → CONSENSUS_REPORT
    → END
"""

from __future__ import annotations
import os
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

# Hotfix for PyTorch / numexpr OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from agents.protocol import AgentID, PipelineState

from agents.agent1_preprocessor import PreprocessorPrime
from agents.agent2_polsar        import PolsarDetective
from agents.agent3_depth         import DepthSounder
from agents.agent4_thermo import ThermoGuardian
from agents.agent5_volume import VolumeOracle
from agents.agent6_terrain import TerrainScout
from agents.agent7_isru import ISRUArchitect
from agents.agent8_navigator import Navigator

log = logging.getLogger("PRISM.ORCHESTRATOR")


# ───────────────────────────────────────────────────────────────────────────
# Configuration loader
# ───────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/pipeline_config.json") -> Dict[str, Any]:
    default = {
        "dfsar_slc_path":      None,
        "dfsar_metadata_path": None,
        "dem_path":            None,
        "ohrc_path":           None,
        "diviner_path":        None,
        "shadowcam_path":      None,
        "illumination_path":   None,
        "layover_threshold":   0.3,
        "output_dir":          "data/outputs",
    }
    p = Path(config_path)
    if p.exists():
        with open(p) as f:
            user = json.load(f)
        default.update(user)
        log.info("Config loaded from %s", config_path)
    else:
        log.warning("Config not found at %s — using defaults (synthetic data mode).", config_path)
    return default


# ───────────────────────────────────────────────────────────────────────────
# Sequential runner (always available)
# ───────────────────────────────────────────────────────────────────────────

def run_sequential(config: Dict[str, Any]) -> PipelineState:
    """
    Guaranteed to work with zero extra dependencies.
    Runs agents in serial; respects data dependencies.
    """
    out_dir = config.get("output_dir", "data/outputs")
    state   = PipelineState()

    a1  = PreprocessorPrime(config, out_dir)
    a2  = PolsarDetective  (config, out_dir)
    a3  = DepthSounder     (config, out_dir)
    a4  = ThermoGuardian   (config, out_dir)
    a5  = VolumeOracle     (config, out_dir)
    a6  = TerrainScout     (config, out_dir)
    a7  = ISRUArchitect    (config, out_dir)
    a8  = Navigator        (config, out_dir)

    # Stage 1
    state = a1.run(state)
    # Stage 2 (logically parallel — serial here)
    state = a2.run(state)
    state = a4.run(state)
    state = a6.run_pass1(state)
    # Stage 3
    state = a3.run(state)
    # Stage 4
    state = a5.run(state)
    # Stage 5
    state = a7.run(state)
    # Stage 6
    state = a6.run_pass2(state)
    # Stage 7
    state = a8.run(state)

    _save_consensus(state, out_dir)
    return state


# ───────────────────────────────────────────────────────────────────────────
# LangGraph runner (optional)
# ───────────────────────────────────────────────────────────────────────────

def run_langgraph(config: Dict[str, Any]) -> PipelineState:
    """
    Uses LangGraph for the DAG with parallel fan-out.
    Falls back to run_sequential if LangGraph is not installed.
    """
    try:
        from langgraph.graph import StateGraph, END as LG_END

        out_dir = config.get("output_dir", "data/outputs")

        a1  = PreprocessorPrime(config, out_dir)
        a2  = PolsarDetective  (config, out_dir)
        a3  = DepthSounder     (config, out_dir)
        a4  = ThermoGuardian   (config, out_dir)
        a5  = VolumeOracle     (config, out_dir)
        a6  = TerrainScout     (config, out_dir)
        a7  = ISRUArchitect    (config, out_dir)
        a8  = Navigator        (config, out_dir)

        # LangGraph nodes must be callables that take state and return state
        def node_a1(s: PipelineState) -> PipelineState: return a1.run(s)
        def node_a2(s: PipelineState) -> PipelineState: return a2.run(s)
        def node_a3(s: PipelineState) -> PipelineState: return a3.run(s)
        def node_a4(s: PipelineState) -> PipelineState: return a4.run(s)
        def node_a5(s: PipelineState) -> PipelineState: return a5.run(s)
        def node_a6p1(s: PipelineState) -> PipelineState: return a6.run_pass1(s)
        def node_a7(s: PipelineState) -> PipelineState: return a7.run(s)
        def node_a6p2(s: PipelineState) -> PipelineState: return a6.run_pass2(s)
        def node_a8(s: PipelineState) -> PipelineState: return a8.run(s)
        def node_consensus(s: PipelineState) -> PipelineState:
            _save_consensus(s, out_dir)
            return s

        g = StateGraph(PipelineState)

        # Register nodes
        for name, fn in [
            ("preprocess",   node_a1),
            ("polsar",       node_a2),
            ("thermo",       node_a4),
            ("terrain_pass1",node_a6p1),
            ("depth",        node_a3),
            ("volume",       node_a5),
            ("isru",         node_a7),
            ("terrain_pass2",node_a6p2),
            ("navigator",    node_a8),
            ("consensus",    node_consensus),
        ]:
            g.add_node(name, fn)

        # Edges (serial for simplicity; LangGraph parallel fan-out is v0.2+)
        g.set_entry_point("preprocess")
        g.add_edge("preprocess",    "polsar")
        g.add_edge("preprocess",    "thermo")
        g.add_edge("preprocess",    "terrain_pass1")
        g.add_edge("polsar",        "depth")
        g.add_edge("depth",         "volume")
        g.add_edge("volume",        "isru")
        g.add_edge("thermo",        "isru")
        g.add_edge("terrain_pass1", "isru")
        g.add_edge("isru",          "terrain_pass2")
        g.add_edge("terrain_pass2", "navigator")
        g.add_edge("navigator",     "consensus")
        g.add_edge("consensus",     LG_END)

        app    = g.compile()
        state  = PipelineState()
        result = app.invoke(state)
        return result

    except ImportError:
        log.warning("LangGraph not installed — falling back to sequential runner.")
        return run_sequential(config)


# ───────────────────────────────────────────────────────────────────────────
# Consensus report
# ───────────────────────────────────────────────────────────────────────────

def _save_consensus(state: PipelineState, out_dir: str) -> None:
    report = {
        "pipeline_complete":      state.pipeline_complete,
        "human_review_required":  state.human_review_required,
        "confidence_registry":    state.confidence_registry,
        "conflict_log":           state.conflict_log,
        "errors":                 state.errors,
        "volume_result":          state.volume_result,
        "landing_sites":          state.landing_sites,
        "traverse_paths":         state.traverse_paths,
        "recommended_path_idx":   state.recommended_path_idx,
        "sensitivity_report":     state.sensitivity_report,
        "output_files": {
            "coregistered_stack":   state.coregistered_stack,
            "quality_mask":         state.quality_mask,
            "cpr_l":                state.cpr_l_path,
            "cpr_s":                state.cpr_s_path,
            "p_ice":                state.p_ice_path,
            "ice_level0":           state.ice_level0_path,
            "boulder_flag":         state.boulder_flag_path,
            "depth_class":          state.depth_class_path,
            "depth_uncertainty":    state.depth_uncertainty_path,
            "ts_raster":            state.ts_raster_path,
            "cold_trap":            state.cold_trap_path,
            "dielectric":           state.dielectric_path,
            "ice_fraction":         state.ice_fraction_path,
            "slope":                state.slope_path,
            "roughness":            state.roughness_path,
            "boulder_density":      state.boulder_density_path,
            "solar_illum":          state.solar_illum_path,
            "ei_raster":            state.ei_path,
            "isru_priority":        state.isru_priority_path,
        },
    }
    out_path = Path(out_dir) / "consensus_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Consensus report → %s", out_path)


# ───────────────────────────────────────────────────────────────────────────
# Main entry point
# ───────────────────────────────────────────────────────────────────────────

def run_pipeline(
    config_path: str = "config/pipeline_config.json",
    use_langgraph: bool = True,
) -> PipelineState:
    """
    Main entry point.

    Args:
        config_path:   Path to pipeline_config.json
        use_langgraph: Try LangGraph DAG first; fall back to sequential if unavailable.

    Returns:
        Completed PipelineState with all outputs populated.
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    config = load_config(config_path)
    t0     = time.time()

    log.info("╔══════════════════════════════════════════╗")
    log.info("║  PRISM — Polarimetric Resource           ║")
    log.info("║  Intelligence System for the Moon        ║")
    log.info("╚══════════════════════════════════════════╝")

    if use_langgraph:
        state = run_langgraph(config)
    else:
        state = run_sequential(config)

    elapsed = time.time() - t0
    log.info("Pipeline finished in %.1f s", elapsed)
    log.info("Errors: %d | Conflicts: %d | Human review: %s",
             len(state.errors), len(state.conflict_log), state.human_review_required)
    return state


if __name__ == "__main__":
    import sys
    cfg  = sys.argv[1] if len(sys.argv) > 1 else "config/pipeline_config.json"
    mode = "--sequential" not in sys.argv
    s    = run_pipeline(cfg, use_langgraph=mode)
    print("\n=== FINAL STATE ===")
    if s.volume_result:
        vr = s.volume_result
        print(f"Total ice (IWE, median): {vr.get('total_ice_iwe_tonnes',{}).get('median','N/A')} tonnes")
    if s.landing_sites:
        top = s.landing_sites[0]
        print(f"Top landing site LS_score: {top.get('ls_score_pass2') or top.get('ls_score_pass1','N/A')}")
    if s.traverse_paths:
        idx = s.recommended_path_idx or 0
        p   = s.traverse_paths[idx]
        print(f"Recommended path: {p['label']} | {p['total_length_m']}m | energy margin {p['energy_margin_pct']:.1f}%")