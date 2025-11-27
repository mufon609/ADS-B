import dataclasses
from typing import Any, Dict

@dataclasses.dataclass
class CaptureJob:
    icao: str
    exposure: float
    seq_log: Dict[str, Any]
    captures_taken: int
    status_payload_base: Dict[str, Any]
