import pandas as pd
from typing import Any, Dict, List

class FlightRecorder:
    def __init__(self, path: str = None, events_path: str = None):
        self.path = path
        self.events_path = events_path
        self._rows: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []

    def log(self, **kwargs):
        self._rows.append(dict(kwargs))

    def log_event(self, run_id: str, event: str, payload: dict, t_wall_ms: float = None):
        import time
        if t_wall_ms is None:
            t_wall_ms = (time.time() * 1000.0)
        self._events.append({
            "run_id": run_id,
            "t_wall_ms": t_wall_ms,
            "event": event,
            "payload": payload
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def events_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._events)

    def flush(self):
        if self.path:
            df = self.to_dataframe()
            df.to_parquet(self.path, index=False)
        if self.events_path:
            edf = self.events_dataframe()
            edf.to_parquet(self.events_path, index=False)
