"""
PRISM — Core Agent Communication Protocol
==========================================
Defines the canonical message format, shared state schema,
and conflict-resolution enums used by every agent in the MAS.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentID(str, Enum):
    PREPROCESSOR_PRIME  = "PREPROCESSOR_PRIME"
    POLSAR_DETECTIVE    = "POLSAR_DETECTIVE"
    DEPTH_SOUNDER       = "DEPTH_SOUNDER"
    THERMO_GUARDIAN     = "THERMO_GUARDIAN"
    VOLUME_ORACLE       = "VOLUME_ORACLE"
    TERRAIN_SCOUT       = "TERRAIN_SCOUT"
    ISRU_ARCHITECT      = "ISRU_ARCHITECT"
    NAVIGATOR           = "NAVIGATOR"
    ORCHESTRATOR        = "ORCHESTRATOR"

class PayloadType(str, Enum):
    RASTER_REFERENCE  = "raster_reference"    # path to a GeoTIFF
    JSON_RESULT       = "json_result"          # structured dict
    STATUS_UPDATE     = "status_update"        # progress ping
    CONFLICT_FLAG     = "conflict_flag"        # inter-agent disagreement
    CONSENSUS_REPORT  = "consensus_report"     # final MAS output

class ConflictLevel(int, Enum):
    NONE    = 0   # < 20 % spatial mismatch — log, proceed
    MINOR   = 1   # 20-50 % — both agents re-examine, apply 0.5× confidence weight
    MAJOR   = 2   # > 50 % or physical impossibility — exclude region, human flag


# ---------------------------------------------------------------------------
# Agent Message
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """
    Canonical inter-agent message.  Every agent sends and receives this.

    JSON wire format
    ----------------
    {
        "sender_agent":    "POLSAR_DETECTIVE",
        "recipient_agent": "DEPTH_SOUNDER",
        "timestamp":       1718700000.0,
        "payload_type":    "raster_reference",
        "agent_confidence": 0.81,
        "payload": {
            "file":           "data/outputs/P_ice_map.tif",
            "crs":            "EPSG:104903",
            "resolution_m":   4.5,
            "quality_mask":   "data/outputs/quality_mask.tif",
            "notes":          "Compact-pol mode; m-chi decomposition used."
        }
    }
    """
    sender_agent:      AgentID
    recipient_agent:   AgentID
    payload_type:      PayloadType
    payload:           Dict[str, Any]
    agent_confidence:  float          = 1.0   # 0.0 – 1.0
    timestamp:         float          = field(default_factory=time.time)
    notes:             str            = ""

    # --- serialisation helpers -------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sender_agent"]    = self.sender_agent.value
        d["recipient_agent"] = self.recipient_agent.value
        d["payload_type"]    = self.payload_type.value
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentMessage":
        return cls(
            sender_agent     = AgentID(d["sender_agent"]),
            recipient_agent  = AgentID(d["recipient_agent"]),
            payload_type     = PayloadType(d["payload_type"]),
            payload          = d["payload"],
            agent_confidence = d.get("agent_confidence", 1.0),
            timestamp        = d.get("timestamp", time.time()),
            notes            = d.get("notes", ""),
        )


# ---------------------------------------------------------------------------
# Pipeline State — the single shared object passed through LangGraph nodes
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """
    Mutable state object that flows through the LangGraph DAG.
    Each agent reads from it and writes its outputs back.
    """
    # ---- data paths (set by agents as they produce outputs) ----------------
    coregistered_stack:    Optional[str] = None   # GeoTIFF
    quality_mask:          Optional[str] = None   # GeoTIFF
    polarization_mode:     Optional[str] = None   # "full_pol" | "compact_pol"
    enl:                   Optional[float] = None  # Equivalent Number of Looks

    cpr_l_path:            Optional[str] = None   # GeoTIFF
    cpr_s_path:            Optional[str] = None   # GeoTIFF
    dop_l_path:            Optional[str] = None   # GeoTIFF
    dop_s_path:            Optional[str] = None   # GeoTIFF
    vsf_path:              Optional[str] = None   # GeoTIFF
    p_ice_path:            Optional[str] = None   # GeoTIFF  0-1
    ice_level0_path:       Optional[str] = None   # GeoTIFF  binary
    boulder_flag_path:     Optional[str] = None   # GeoTIFF  binary

    depth_class_path:      Optional[str] = None   # GeoTIFF  0/1/2
    depth_uncertainty_path:Optional[str] = None   # GeoTIFF
    p_shallow_path:        Optional[str] = None   # GeoTIFF
    p_deep_path:           Optional[str] = None   # GeoTIFF

    ts_raster_path:        Optional[str] = None   # GeoTIFF  thermal stability
    cold_trap_path:        Optional[str] = None   # GeoTIFF  cold trap class

    dielectric_path:       Optional[str] = None   # GeoTIFF
    ice_fraction_path:     Optional[str] = None   # GeoTIFF
    volume_result:         Optional[Dict] = None   # see VolumeResult below

    slope_path:            Optional[str] = None   # GeoTIFF
    roughness_path:        Optional[str] = None   # GeoTIFF
    boulder_density_path:  Optional[str] = None   # GeoTIFF
    solar_illum_path:      Optional[str] = None   # GeoTIFF
    landing_sites:         Optional[List[Dict]] = None  # top-3 dicts

    ei_path:               Optional[str] = None   # GeoTIFF  0-1
    isru_priority_path:    Optional[str] = None   # GeoTIFF  3-class
    sensitivity_report:    Optional[Dict] = None

    traverse_paths:        Optional[List[Dict]] = None  # list of path dicts
    recommended_path_idx:  Optional[int]  = None

    # ---- inter-agent confidence registry ----------------------------------
    confidence_registry:   Dict[str, float] = field(default_factory=dict)
    # key = AgentID.value, value = 0-1

    # ---- conflict log ------------------------------------------------------
    conflict_log:          List[Dict] = field(default_factory=list)

    # ---- message bus (append-only, for audit) ------------------------------
    message_bus:           List[Dict] = field(default_factory=list)

    # ---- runtime flags -----------------------------------------------------
    pipeline_complete:     bool  = False
    human_review_required: bool  = False
    errors:                List[str] = field(default_factory=list)

    def post_message(self, msg: AgentMessage) -> None:
        """Append to the immutable audit log."""
        self.message_bus.append(msg.to_dict())

    def register_confidence(self, agent: AgentID, confidence: float) -> None:
        self.confidence_registry[agent.value] = max(0.0, min(1.0, confidence))

    def log_conflict(
        self,
        agent_a: AgentID,
        agent_b: AgentID,
        level: ConflictLevel,
        description: str,
        resolution: str = "",
    ) -> None:
        self.conflict_log.append({
            "agent_a":     agent_a.value,
            "agent_b":     agent_b.value,
            "level":       level.value,
            "description": description,
            "resolution":  resolution,
            "timestamp":   time.time(),
        })
        if level == ConflictLevel.MAJOR:
            self.human_review_required = True


# ---------------------------------------------------------------------------
# Canonical output schemas (docstrings = the contract each agent must honour)
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorOutput:
    """
    Reported by: PREPROCESSOR_PRIME
    Required keys in PipelineState after this agent runs:
        coregistered_stack, quality_mask, polarization_mode, enl
    JSON quality report schema:
    {
        "polarization_mode": "full_pol",           # or "compact_pol"
        "enl": 12.4,
        "nesz_db": {"L": -22.1, "S": -19.8},
        "layover_fraction": 0.03,
        "shadow_fraction": 0.08,
        "co_registration_rmse_px": 0.4,
        "bands_available": ["HH","HV","VH","VV"]   # or ["RH","RV"]
    }
    """
    pass


@dataclass
class PolsarOutput:
    """
    Reported by: POLSAR_DETECTIVE
    GeoTIFFs: cpr_l, cpr_s, dop_l, dop_s, vsf, p_ice, ice_level0, boulder_flag
    JSON confidence payload:
    {
        "agent_confidence": 0.81,
        "rf_oob_accuracy": 0.88,
        "rf_spatial_cv_auc": 0.86,
        "ice_positive_pixels": 14230,
        "boulder_false_positive_pixels": 1840,
        "decomposition_used": "m_chi",    # or "yamaguchi"
        "notes": "compact_pol mode; Yamaguchi not applicable"
    }
    """
    pass


@dataclass
class DepthSounderOutput:
    """
    Reported by: DEPTH_SOUNDER
    GeoTIFFs: depth_class (0=none, 1=shallow 0-2m, 2=deep 2-5m),
              depth_uncertainty, p_shallow, p_deep
    JSON confidence payload:
    {
        "agent_confidence": 0.76,
        "shallow_pixel_count": 8200,
        "deep_pixel_count": 3100,
        "uncertain_pixel_count": 900,   # max(P) < 0.6
        "mc_samples_used": 500,
        "cross_validation_with_idea_c": "pending"   # or "agreement" / "disagreement"
    }
    """
    pass


@dataclass
class ThermoGuardianOutput:
    """
    Reported by: THERMO_GUARDIAN
    GeoTIFFs: ts_raster (0-1), cold_trap_class (0=none,1=cold,2=extreme,3=super)
    JSON confidence payload:
    {
        "agent_confidence": 0.90,
        "super_cold_trap_pixels": 5400,
        "diviner_interp_method": "bilinear",
        "misregistration_flagged_pixels": 120,
        "volatile_age_gyr": 1.2,     # Idea A — null if ShadowCam unavailable
        "volatile_age_uncertainty_gyr": 0.4
    }
    """
    pass


@dataclass
class VolumeOracleOutput:
    """
    Reported by: VOLUME_ORACLE
    GeoTIFFs: dielectric_constant, ice_fraction_f
    JSON volume_result:
    {
        "agent_confidence": 0.74,
        "total_ice_volume_m3": {"median": 1.2e6, "p5": 4e5, "p95": 2.8e6},
        "total_ice_iwe_tonnes": {"median": 1.1e6, "p5": 3.7e5, "p95": 2.6e6},
        "extractable_ice_iwe_tonnes": {"median": 6.5e5, "p5": 2.1e5, "p95": 1.5e6},
        "mc_runs": 1000,
        "inversion_method_primary": "IEM",
        "inversion_method_backup": "Oh2004",
        "flagged_pixels_divergence": 340,
        "mean_ice_fraction_f": 0.12
    }
    """
    pass


@dataclass
class TerrainScoutOutput:
    """
    Reported by: TERRAIN_SCOUT
    GeoTIFFs: slope, roughness, boulder_density, solar_illum
    JSON landing_sites (list of top-3):
    [
      {
        "rank": 1,
        "lon_deg": 87.23,
        "lat_deg": -87.45,
        "slope_deg": 4.2,
        "roughness_cm": 8.1,
        "solar_fraction": 0.62,
        "dist_to_high_ei_m": 320,
        "ls_score_pass1": 0.71,
        "ls_score_pass2": 0.74,
        "los_to_earth": true,
        "justification": "Low slope, near high-EI zone, good solar"
      }
    ]
    """
    pass


@dataclass
class ISRUArchitectOutput:
    """
    Reported by: ISRU_ARCHITECT
    GeoTIFFs: ei_raster (0-1), isru_priority (0=low,1=med,2=high)
    JSON sensitivity_report:
    {
        "agent_confidence": 0.85,
        "mean_rank_correlation": 0.91,
        "ranking_robust": true,
        "baseline_weights": {"w1_depth": 0.4, "w2_compaction": 0.3, "w3_thermal": 0.3},
        "weight_sweep_n": 125,
        "extractable_volume_iwe": {"median": 6.5e5, "ci90_low": 2.1e5, "ci90_high": 1.5e6},
        "conflict_checks": {
            "agent2_vs_agent5_overlap": 0.78,
            "agent3_vs_ideaC_agreement": 0.72,
            "agent4_vs_agent6_TS_override_pixels": 0,
            "EI_vs_illumination_spearman_rho": -0.67
        }
    }
    """
    pass


@dataclass
class NavigatorOutput:
    """
    Reported by: NAVIGATOR
    JSON traverse_paths (list of 3):
    [
      {
        "path_id": 0,
        "label": "min_terrain",
        "waypoints": [[lon1,lat1], [lon2,lat2], ...],  # decimal degrees
        "total_length_m": 2340,
        "terrain_risk_score": 0.23,
        "solar_feasibility_pct": 88,
        "science_score_ei_sum": 4.2,
        "energy_consumed_wh": 68.4,
        "energy_available_wh": 82.1,
        "energy_margin_pct": 20.0,
        "charging_waypoints": [[lon,lat], ...],
        "feasible": true
      }
    ]
    recommended_path_idx: 2   (the balanced Pareto path)
    """
    pass