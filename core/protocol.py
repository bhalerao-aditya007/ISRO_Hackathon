"""
PRISM — Core Agent Communication Protocol
==========================================
Defines the canonical message format, shared state schema,
and conflict-resolution enums used by every agent in the MAS.

CHANGES FROM ORIGINAL:
  - Added JSON serialisation safety (default=str) in to_dict for Path objects.
  - PipelineState is now JSON-serialisable for the REST API.
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
    RASTER_REFERENCE  = "raster_reference"
    JSON_RESULT       = "json_result"
    STATUS_UPDATE     = "status_update"
    CONFLICT_FLAG     = "conflict_flag"
    CONSENSUS_REPORT  = "consensus_report"

class ConflictLevel(int, Enum):
    NONE    = 0
    MINOR   = 1
    MAJOR   = 2


# ---------------------------------------------------------------------------
# Agent Message
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    sender_agent:      AgentID
    recipient_agent:   AgentID
    payload_type:      PayloadType
    payload:           Dict[str, Any]
    agent_confidence:  float          = 1.0
    timestamp:         float          = field(default_factory=time.time)
    notes:             str            = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sender_agent"]    = self.sender_agent.value
        d["recipient_agent"] = self.recipient_agent.value
        d["payload_type"]    = self.payload_type.value
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

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
# Pipeline State
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """
    Mutable state object that flows through the LangGraph DAG.
    Each agent reads from it and writes its outputs back.
    """
    coregistered_stack:    Optional[str] = None
    quality_mask:          Optional[str] = None
    polarization_mode:     Optional[str] = None
    enl:                   Optional[float] = None

    cpr_l_path:            Optional[str] = None
    cpr_s_path:            Optional[str] = None
    dop_l_path:            Optional[str] = None
    dop_s_path:            Optional[str] = None
    vsf_path:              Optional[str] = None
    p_ice_path:            Optional[str] = None
    ice_level0_path:       Optional[str] = None
    boulder_flag_path:     Optional[str] = None

    depth_class_path:      Optional[str] = None
    depth_uncertainty_path:Optional[str] = None
    p_shallow_path:        Optional[str] = None
    p_deep_path:           Optional[str] = None

    ts_raster_path:        Optional[str] = None
    cold_trap_path:        Optional[str] = None

    dielectric_path:       Optional[str] = None
    ice_fraction_path:     Optional[str] = None
    volume_result:         Optional[Dict] = None

    slope_path:            Optional[str] = None
    roughness_path:        Optional[str] = None
    boulder_density_path:  Optional[str] = None
    solar_illum_path:      Optional[str] = None
    landing_sites:         Optional[List[Dict]] = None

    ei_path:               Optional[str] = None
    isru_priority_path:    Optional[str] = None
    sensitivity_report:    Optional[Dict] = None

    traverse_paths:        Optional[List[Dict]] = None
    recommended_path_idx:  Optional[int]  = None

    confidence_registry:   Dict[str, float] = field(default_factory=dict)
    conflict_log:          List[Dict] = field(default_factory=list)
    message_bus:           List[Dict] = field(default_factory=list)

    pipeline_complete:     bool  = False
    human_review_required: bool  = False
    errors:                List[str] = field(default_factory=list)

    def post_message(self, msg: AgentMessage) -> None:
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

    def to_summary_dict(self) -> Dict[str, Any]:
        """Lightweight dict safe to return from the REST API."""
        return {
            "pipeline_complete":     self.pipeline_complete,
            "human_review_required": self.human_review_required,
            "polarization_mode":     self.polarization_mode,
            "enl":                   self.enl,
            "confidence_registry":   self.confidence_registry,
            "conflict_count":        len(self.conflict_log),
            "error_count":           len(self.errors),
            "errors":                self.errors,
            "volume_result":         self.volume_result,
            "landing_sites":         self.landing_sites,
            "traverse_paths":        self.traverse_paths,
            "recommended_path_idx":  self.recommended_path_idx,
            "sensitivity_report":    self.sensitivity_report,
            "output_files": {
                "coregistered_stack":  self.coregistered_stack,
                "p_ice":               self.p_ice_path,
                "depth_class":         self.depth_class_path,
                "ts_raster":           self.ts_raster_path,
                "cold_trap":           self.cold_trap_path,
                "ei_raster":           self.ei_path,
                "isru_priority":       self.isru_priority_path,
                "slope":               self.slope_path,
            },
        }


# ---------------------------------------------------------------------------
# Canonical output schemas (unchanged from original)
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorOutput:
    pass

@dataclass
class PolsarOutput:
    pass

@dataclass
class DepthSounderOutput:
    pass

@dataclass
class ThermoGuardianOutput:
    pass

@dataclass
class VolumeOracleOutput:
    pass

@dataclass
class TerrainScoutOutput:
    pass

@dataclass
class ISRUArchitectOutput:
    pass

@dataclass
class NavigatorOutput:
    pass
