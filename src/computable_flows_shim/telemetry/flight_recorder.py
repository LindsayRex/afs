import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any, Dict, List, Optional

class FlightRecorder:
    def __init__(self, path: Optional[str] = None, events_path: Optional[str] = None):
        self.path = path
        self.events_path = events_path
        self._rows: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []

    def log(self, **kwargs):
        self._rows.append(dict(kwargs))

    def log_event(self, run_id: str, event: str, payload: dict, t_wall_ms: Optional[float] = None):
        import time
        if t_wall_ms is None:
            t_wall_ms = (time.time() * 1000.0)
        self._events.append({
            "run_id": run_id,
            "t_wall_ms": t_wall_ms,
            "event": event,
            "payload": str(payload)  # Parquet doesn't handle nested dicts well
        })

    def to_arrow_table(self, records: List[Dict[str, Any]]) -> pa.Table:
        if not records:
            return pa.Table.from_pylist([])
        return pa.Table.from_pylist(records)

    def flush(self):
        if self.path and self._rows:
            table = self.to_arrow_table(self._rows)
            pq.write_table(table, self.path)
        if self.events_path and self._events:
            events_table = self.to_arrow_table(self._events)
            pq.write_table(events_table, self.events_path)
