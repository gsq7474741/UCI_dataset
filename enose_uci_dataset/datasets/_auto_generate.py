"""
Auto-regenerate _generated.py if YAML schemas are newer.

This module is imported before _generated.py to ensure the generated code
is always up-to-date with the YAML schemas.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Paths
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
_SCHEMAS_DIR = _PROJECT_ROOT / "schemas"
_GENERATED_FILE = _THIS_DIR / "_generated.py"
_SCHEMA_FILES = ["sensors.yaml", "datasets.yaml", "labels.yaml"]


def _needs_regeneration() -> bool:
    """Check if _generated.py needs to be regenerated.
    
    Returns True if:
    - _generated.py doesn't exist
    - Any schema file is newer than _generated.py
    """
    if not _GENERATED_FILE.exists():
        return True
    
    generated_mtime = _GENERATED_FILE.stat().st_mtime
    
    for schema_file in _SCHEMA_FILES:
        schema_path = _SCHEMAS_DIR / schema_file
        if schema_path.exists():
            if schema_path.stat().st_mtime > generated_mtime:
                return True
    
    return False


def _regenerate() -> bool:
    """Regenerate _generated.py from YAML schemas.
    
    Returns True if successful, False otherwise.
    """
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed, cannot auto-regenerate metadata", file=sys.stderr)
        return False
    
    if not _SCHEMAS_DIR.exists():
        return False
    
    from datetime import datetime
    from typing import Any, Dict, List
    
    def load_yaml(filename: str) -> Dict[str, Any]:
        path = _SCHEMAS_DIR / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def generate_header() -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f'''"""
AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
============================================
Generated from YAML schemas in schemas/ directory.
Timestamp: {timestamp}

To modify sensor/dataset definitions, edit the YAML files:
    - schemas/sensors.yaml
    - schemas/datasets.yaml
    - schemas/labels.yaml

This file is auto-regenerated on import if schemas change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

'''

    def generate_sensor_models(sensors: List[Dict]) -> str:
        lines = [
            "# =============================================================================",
            "# SENSOR MODELS (from schemas/sensors.yaml)",
            "# =============================================================================",
            "",
            "@dataclass(frozen=True)",
            "class SensorModel:",
            '    """Global sensor model definition with target gas mapping."""',
            "    id: int",
            "    name: str",
            "    target_gases: FrozenSet[str]",
            "    sensor_type: str",
            "    manufacturer: str",
            '    description: str = ""',
            "",
            "",
            "SENSOR_MODELS: Tuple[SensorModel, ...] = (",
        ]
        
        for sensor in sensors:
            gases = ", ".join(f'"{g}"' for g in sensor.get("target_gases", []))
            lines.append(
                f'    SensorModel({sensor["id"]}, "{sensor["name"]}", '
                f'frozenset({{{gases}}}), "{sensor["type"]}", '
                f'"{sensor["manufacturer"]}", "{sensor.get("description", "")}"),'
            )
        
        lines.extend([
            ")",
            "",
            "# Quick lookup: sensor name -> global index",
            "SENSOR_NAME_TO_ID: Dict[str, int] = {s.name: s.id for s in SENSOR_MODELS}",
            "",
            "# Total dimension of universal sensor space",
            "M_TOTAL: int = len(SENSOR_MODELS)",
            "",
            "",
            "def get_sensor_id(name: str) -> int:",
            '    """Get global sensor index by name. Returns 0 (UNKNOWN) if not found."""',
            "    return SENSOR_NAME_TO_ID.get(name, 0)",
            "",
            "",
            "def get_sensor_model(name: str) -> SensorModel:",
            '    """Get sensor model by name. Returns UNKNOWN if not found."""',
            "    idx = get_sensor_id(name)",
            "    return SENSOR_MODELS[idx]",
            "",
        ])
        
        return "\n".join(lines)

    def generate_dataset_metadata(datasets: List[Dict]) -> str:
        lines = [
            "# =============================================================================",
            "# DATASET METADATA (from schemas/datasets.yaml)",
            "# =============================================================================",
            "",
            "DATASET_SAMPLE_RATES: Dict[str, Optional[float]] = {",
        ]
        
        for ds in datasets:
            rate = ds.get("sample_rate_hz")
            rate_str = str(rate) if rate is not None else "None"
            lines.append(f'    "{ds["name"]}": {rate_str},')
        
        lines.extend([
            "}",
            "",
            "DATASET_COLLECTION_TYPE: Dict[str, str] = {",
        ])
        
        for ds in datasets:
            lines.append(f'    "{ds["name"]}": "{ds["collection_type"]}",')
        
        lines.extend([
            "}",
            "",
            "DATASET_RESPONSE_TYPE: Dict[str, str] = {",
        ])
        
        for ds in datasets:
            lines.append(f'    "{ds["name"]}": "{ds["response_type"]}",')
        
        lines.extend([
            "}",
            "",
            "# Channel to global sensor ID mapping",
            "DATASET_CHANNEL_TO_GLOBAL: Dict[str, List[int]] = {",
        ])
        
        for ds in datasets:
            channel_ids = [f'get_sensor_id("{ch}")' for ch in ds.get("channels", [])]
            if len(channel_ids) <= 4:
                channels_str = ", ".join(channel_ids)
                lines.append(f'    "{ds["name"]}": [{channels_str}],')
            else:
                lines.append(f'    "{ds["name"]}": [')
                for ch_id in channel_ids:
                    lines.append(f'        {ch_id},')
                lines.append("    ],")
        
        lines.extend([
            "}",
            "",
            "",
            "def get_global_channel_mapping(dataset_name: str) -> List[int]:",
            '    """Get global sensor indices for a dataset\'s channels."""',
            "    return DATASET_CHANNEL_TO_GLOBAL.get(dataset_name, [])",
            "",
        ])
        
        return "\n".join(lines)

    def generate_gas_labels(labels_data: Dict) -> str:
        gas_labels = labels_data.get("gas_labels", [])
        mappings = labels_data.get("gas_label_mappings", {})
        
        lines = [
            "# =============================================================================",
            "# GAS LABELS (from schemas/labels.yaml)",
            "# =============================================================================",
            "",
            "# Gas label name to ID mapping",
            "GAS_LABEL_TO_ID: Dict[str, int] = {",
        ]
        
        for label in gas_labels:
            lines.append(f'    "{label["name"]}": {label["id"]},')
        
        lines.extend([
            "}",
            "",
            "# ID to gas label name mapping",
            "GAS_LABEL_ID_TO_NAME: Dict[int, str] = {v: k for k, v in GAS_LABEL_TO_ID.items()}",
            "",
            "# Dataset-local to global label mappings",
            "GAS_LABEL_MAPPINGS: Dict[str, Dict[int, int]] = {",
        ])
        
        for ds_name, ds_mappings in mappings.items():
            lines.append(f'    "{ds_name}": {{')
            for local_id, global_name in ds_mappings.items():
                global_id = next((l["id"] for l in gas_labels if l["name"] == global_name), 0)
                lines.append(f'        {local_id}: {global_id},  # {global_name}')
            lines.append("    },")
        
        lines.extend([
            "}",
            "",
            "",
            "def get_global_gas_label(dataset_name: str, local_label: int) -> int:",
            '    """Convert dataset-local label to global label index."""',
            "    mapping = GAS_LABEL_MAPPINGS.get(dataset_name, {})",
            "    return mapping.get(local_label, 0)  # 0 = UNKNOWN",
            "",
        ])
        
        return "\n".join(lines)

    # Load schemas
    sensors_data = load_yaml("sensors.yaml")
    datasets_data = load_yaml("datasets.yaml")
    labels_data = load_yaml("labels.yaml")
    
    if not sensors_data.get("sensors") or not datasets_data.get("datasets"):
        return False
    
    # Generate code
    code_parts = [
        generate_header(),
        generate_sensor_models(sensors_data["sensors"]),
        generate_dataset_metadata(datasets_data["datasets"]),
        generate_gas_labels(labels_data),
    ]
    
    code = "\n".join(code_parts)
    
    # Write output
    with open(_GENERATED_FILE, "w", encoding="utf-8") as f:
        f.write(code)
    
    return True


def ensure_generated():
    """Ensure _generated.py is up-to-date with YAML schemas."""
    if _needs_regeneration():
        if _regenerate():
            # Clear any cached imports
            module_name = "enose_uci_dataset.datasets._generated"
            if module_name in sys.modules:
                del sys.modules[module_name]


# Auto-run on import
ensure_generated()
