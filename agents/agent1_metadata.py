"""
agent1_metadata.py — PREPROCESSOR PRIME, metadata-parsing submodule.

Parses the actual ISRO PDS4-format XML labels shipped alongside DFSAR raw
data and orbit/geometry CSVs. Field names below are taken VERBATIM from a
real label inspected on 2026-06-18:

    ch2_sar_nrxl_20250413t195013259_d_r0b_xx_cp_xx_d18.xml   (data label)
    ch2_sar_nrxl_20250413t195013259_g_oat_xx_cp_xx_d18.xml   (orbit label)

IMPORTANT CORRECTION vs the original project outline:
  - The outline assumed DFSAR ships as "SLC" (already range-compressed,
    single-look complex) data and called the polarization mode "compact-pol"
    (RH/RV). The REAL label says:
        processing_level   = "Raw"        (i.e. L0B-RAW, NOT SLC)
        num_polarizations  = 2
        polarization       = "LH", "LV"   (L-band Horizontal/Vertical receive)
    This is DUAL-POL (single transmit, dual receive), not RH/RV compact-pol.
    Do NOT assume m-chi/Raney-2012 compact-pol decomposition applies as-is —
    that decomposition is specifically for RH/RV hybrid-pol data. For LH/LV
    dual-pol data, the correct baseline is computing the polarimetric ratio
    LH/LV (or individual sigma-naughts) directly; full Stokes-vector
    reconstruction used for hybrid-pol does NOT apply unmodified. Agent 2
    must re-derive which CPR/decomposition formula is valid for LH/LV before
    using Raney (2012) verbatim. Flagging this here so it isn't silently
    assumed correct downstream.
  - Because processing_level == "Raw", a SAR focusing / range-compression
    step is required before any calibrated sigma-naught products exist.
    This is precisely the pyroSAR -> SNAP gpt step in agent1_calibration.py.

This module does ONLY parsing + validation. It does not touch the multi-GB
.dat file's pixel contents (that's a job for the calibration step, since raw
focusing must go through SNAP, not hand-rolled numpy).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

# PDS4 / ISDA namespace map exactly as declared in the real label's root tag.
NS = {
    "pds": "http://pds.nasa.gov/pds4/pds/v1",
    "isda": "http://pds.nasa.gov/pds4/isda/v1",
}


@dataclass
class PolarizationChannel:
    """One <isda:polarization_info> block."""
    polarization: str               # e.g. "LH", "LV"
    bias_real: float
    bias_imag: float
    standard_deviation_real: float
    standard_deviation_imag: float
    gain_imbalance: float
    phase_orthogonality: float
    nes0_coeff_0: float
    nes0_coeff_1: float


@dataclass
class DFSARDataLabel:
    """Parsed contents of the *_d_r0b_xx_*.xml (raw SAR data) label."""
    logical_identifier: str
    processing_level: str           # "Raw" expected for current data drop
    product_type: str                # e.g. "L0B-RAW"
    frequency_band: str               # "L" or "S"
    imaging_mode: str                 # e.g. "STRIPMAP"
    polarization_mode: str            # derived classification, see classify_polarization_mode()
    num_polarizations: int
    polarizations: list[PolarizationChannel]
    incidence_angle_deg: float
    look_angle_deg: float
    look_direction: str
    node: str                         # "ASCENDING" / "DESCENDING"
    pulse_repetition_frequency_hz: float
    radar_center_frequency_hz: float
    samples_per_echo_line: int
    pulses_received_per_dwell: int
    data_type: str                    # element array data type, e.g. "SignedByte"
    array_lines: int
    array_samples: int
    file_name: str
    file_size_bytes: int
    md5_checksum: str
    # Footprint corners (lat, lon) in degrees
    upper_left: tuple[float, float]
    upper_right: tuple[float, float]
    lower_right: tuple[float, float]
    lower_left: tuple[float, float]
    centre: tuple[float, float]
    semi_major_radius_m: float
    semi_minor_radius_m: float
    source_xml_path: str


def classify_polarization_mode(num_pols: int, pol_names: list[str]) -> str:
    """
    Classify the real polarization mode from observed channel names.

    This is the corrected version of the outline's polarization-mode check
    (originally written for "full-pol vs compact-pol"). Real ISRO DFSAR
    labels show channel names like LH/LV (dual-pol, linear H/V receive on
    one transmit polarization) rather than RH/RV (circular, hybrid/compact-pol).
    We classify based on what's actually present rather than assuming.
    """
    names = set(p.upper() for p in pol_names)

    if num_pols == 4 and names.issuperset({"HH", "HV", "VH", "VV"}):
        return "FULL_POL_QUAD"
    if num_pols == 2 and names.issubset({"RH", "RV"}) and names:
        return "COMPACT_POL_HYBRID"  # Raney (2012) m-chi applies directly
    if num_pols == 2 and (names.issubset({"LH", "LV"}) or names.issubset({"HH", "HV"}) or names.issubset({"VH", "VV"})):
        return "DUAL_POL_LINEAR"     # Raney m-chi does NOT apply unmodified — see module docstring
    if num_pols == 1:
        return "SINGLE_POL"
    return f"UNKNOWN_MODE_n{num_pols}_{'_'.join(sorted(names))}"


def _text(elem: Optional[ET.Element], default: str = "") -> str:
    if elem is None or elem.text is None:
        return default
    return elem.text.strip()


def _float(elem: Optional[ET.Element], default: float = float("nan")) -> float:
    txt = _text(elem)
    try:
        return float(txt)
    except ValueError:
        return default


def parse_dfsar_data_label(xml_path: str | Path) -> DFSARDataLabel:
    """
    Parse a DFSAR raw-data PDS4 label (the *_d_r0b_xx_*.xml file).

    Raises FileNotFoundError if the path doesn't exist, and ValueError if
    expected PDS4 structure is missing — we fail loudly rather than
    returning a half-populated object, since downstream agents trust this.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"DFSAR label not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    ident = root.find("pds:Identification_Area", NS)
    if ident is None:
        raise ValueError(f"No Identification_Area found in {xml_path} — unexpected PDS4 structure")

    logical_id = _text(ident.find("pds:logical_identifier", NS))

    obs_area = root.find("pds:Observation_Area", NS)
    summary = obs_area.find("pds:Primary_Result_Summary", NS)
    processing_level = _text(summary.find("pds:processing_level", NS))

    mission_area = obs_area.find("pds:Mission_Area", NS)
    params = mission_area.find("isda:Product_Parameters", NS)
    geom = mission_area.find("isda:Geometry_Parameters", NS)

    product_type = _text(params.find("isda:product_type", NS))
    frequency_band = _text(params.find("isda:frequency_band", NS))
    imaging_mode = _text(params.find("isda:imaging_mode", NS))
    num_pols = int(_text(params.find("isda:num_polarizations", NS), "0") or 0)

    pol_channels: list[PolarizationChannel] = []
    for pol_elem in params.findall("isda:polarization_info", NS):
        pol_channels.append(PolarizationChannel(
            polarization=_text(pol_elem.find("isda:polarization", NS)),
            bias_real=_float(pol_elem.find("isda:bias_real", NS)),
            bias_imag=_float(pol_elem.find("isda:bias_imag", NS)),
            standard_deviation_real=_float(pol_elem.find("isda:standard_deviation_real", NS)),
            standard_deviation_imag=_float(pol_elem.find("isda:standard_deviation_imag", NS)),
            gain_imbalance=_float(pol_elem.find("isda:gain_imbalance", NS)),
            phase_orthogonality=_float(pol_elem.find("isda:phase_orthogonality", NS)),
            nes0_coeff_0=_float(pol_elem.find("isda:nes0_coeff_0", NS)),
            nes0_coeff_1=_float(pol_elem.find("isda:nes0_coeff_1", NS)),
        ))

    pol_mode = classify_polarization_mode(num_pols, [p.polarization for p in pol_channels])

    incidence_angle = _float(params.find("isda:incidence_angle", NS))
    look_angle = _float(params.find("isda:look_angle", NS))
    look_direction = _text(params.find("isda:look_direction", NS))
    node = _text(params.find("isda:node", NS))
    prf = _float(params.find("isda:pulse_repetition_frequency", NS))
    center_freq = _float(params.find("isda:radar_center_frequency", NS))
    samples_per_echo = int(_text(params.find("isda:samples_per_echo_line", NS), "0") or 0)
    pulses_per_dwell = int(_text(params.find("isda:pulses_received_per_dwell", NS), "0") or 0)

    semi_major = _float(params.find("isda:semi_major_radius", NS))
    semi_minor = _float(params.find("isda:semi_minor_radius", NS))

    upper_left = (_float(geom.find("isda:upper_left_latitude", NS)), _float(geom.find("isda:upper_left_longitude", NS)))
    upper_right = (_float(geom.find("isda:upper_right_latitude", NS)), _float(geom.find("isda:upper_right_longitude", NS)))
    lower_right = (_float(geom.find("isda:lower_right_latitude", NS)), _float(geom.find("isda:lower_right_longitude", NS)))
    lower_left = (_float(geom.find("isda:lower_left_latitude", NS)), _float(geom.find("isda:lower_left_longitude", NS)))
    centre = (_float(geom.find("isda:centre_latitude", NS)), _float(geom.find("isda:centre_longitude", NS)))

    file_area = root.find("pds:File_Area_Observational", NS)
    file_elem = file_area.find("pds:File", NS)
    file_name = _text(file_elem.find("pds:file_name", NS))
    file_size = int(_text(file_elem.find("pds:file_size", NS), "0") or 0)
    md5 = _text(file_elem.find("pds:md5_checksum", NS))

    array = file_area.find("pds:Array_2D_Image", NS)
    data_type = _text(array.find("pds:Element_Array", NS).find("pds:data_type", NS))
    axes = array.findall("pds:Axis_Array", NS)
    array_lines, array_samples = 0, 0
    for ax in axes:
        name = _text(ax.find("pds:axis_name", NS))
        n = int(_text(ax.find("pds:elements", NS), "0") or 0)
        if name == "Line":
            array_lines = n
        elif name == "Sample":
            array_samples = n

    return DFSARDataLabel(
        logical_identifier=logical_id,
        processing_level=processing_level,
        product_type=product_type,
        frequency_band=frequency_band,
        imaging_mode=imaging_mode,
        polarization_mode=pol_mode,
        num_polarizations=num_pols,
        polarizations=pol_channels,
        incidence_angle_deg=incidence_angle,
        look_angle_deg=look_angle,
        look_direction=look_direction,
        node=node,
        pulse_repetition_frequency_hz=prf,
        radar_center_frequency_hz=center_freq,
        samples_per_echo_line=samples_per_echo,
        pulses_received_per_dwell=pulses_per_dwell,
        data_type=data_type,
        array_lines=array_lines,
        array_samples=array_samples,
        file_name=file_name,
        file_size_bytes=file_size,
        md5_checksum=md5,
        upper_left=upper_left,
        upper_right=upper_right,
        lower_right=lower_right,
        lower_left=lower_left,
        centre=centre,
        semi_major_radius_m=semi_major,
        semi_minor_radius_m=semi_minor,
        source_xml_path=str(xml_path),
    )


def validate_label(label: DFSARDataLabel) -> list[str]:
    """
    Sanity-check a parsed label against physically/scientifically expected
    ranges. Returns a list of warning strings (empty list = all clear).
    Agent 1 should log these in its quality report; it must NOT silently
    proceed if critical fields are NaN or zero.
    """
    warnings: list[str] = []

    if label.processing_level.lower() != "raw":
        warnings.append(
            f"Unexpected processing_level='{label.processing_level}' "
            f"(expected 'Raw' for current L0B-RAW data drop — pipeline logic "
            f"for calibration step assumes raw input; re-check if this changes)."
        )

    if label.frequency_band not in ("L", "S"):
        warnings.append(f"Unexpected frequency_band='{label.frequency_band}' (expected 'L' or 'S').")

    if label.polarization_mode == "DUAL_POL_LINEAR":
        warnings.append(
            "DUAL_POL_LINEAR detected (e.g. LH/LV). Raney (2012) m-chi "
            "compact-pol decomposition does NOT apply unmodified — Agent 2 "
            "must use the dual-pol-appropriate ratio/feature formulation, "
            "not the hybrid-pol Stokes reconstruction."
        )
    elif label.polarization_mode.startswith("UNKNOWN_MODE"):
        warnings.append(
            f"Polarization mode could not be classified: '{label.polarization_mode}'. "
            f"Manual review required before Agent 2 runs."
        )

    if not (0 <= label.incidence_angle_deg <= 90):
        warnings.append(f"incidence_angle_deg={label.incidence_angle_deg} is outside [0,90] — check parsing.")

    if label.array_lines <= 0 or label.array_samples <= 0:
        warnings.append(f"Array dimensions invalid: lines={label.array_lines}, samples={label.array_samples}.")

    if abs(label.semi_major_radius_m - 1737400) > 1.0:
        warnings.append(
            f"semi_major_radius_m={label.semi_major_radius_m} differs from "
            f"config.CRS reference radius (1737400m) — verify CRS consistency."
        )

    return warnings


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python agent1_metadata.py <path_to_data_label.xml>")
        sys.exit(1)

    label = parse_dfsar_data_label(sys.argv[1])
    warnings = validate_label(label)

    print("=" * 70)
    print(f"Parsed: {label.file_name}")
    print("=" * 70)
    print(f"  processing_level     : {label.processing_level}")
    print(f"  product_type         : {label.product_type}")
    print(f"  frequency_band       : {label.frequency_band}")
    print(f"  imaging_mode         : {label.imaging_mode}")
    print(f"  polarization_mode    : {label.polarization_mode}")
    print(f"  num_polarizations    : {label.num_polarizations}")
    print(f"  channels             : {[p.polarization for p in label.polarizations]}")
    print(f"  incidence_angle_deg  : {label.incidence_angle_deg}")
    print(f"  look_direction/node  : {label.look_direction} / {label.node}")
    print(f"  array (lines x samp) : {label.array_lines} x {label.array_samples}")
    print(f"  data_type            : {label.data_type}")
    print(f"  centre (lat, lon)    : {label.centre}")
    print(f"  file_size_bytes      : {label.file_size_bytes:,}")
    print()
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("No warnings — label passed all sanity checks.")