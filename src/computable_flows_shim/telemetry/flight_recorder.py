import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import os
from typing import Any, Dict, List, Optional

class FlightRecorder:
    def __init__(self, path: Optional[str] = None, events_path: Optional[str] = None, schema_version: int = 3):
        self.path = path
        self.events_path = events_path
        self.schema_version = schema_version
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
            "payload": str(payload)
        })

    def to_arrow_table(self, records: List[Dict[str, Any]]) -> pa.Table:
        if not records:
            return pa.Table.from_pylist([])
        
        # Add schema version to metadata
        metadata = {'schema_version': str(self.schema_version)}
        
        # Create table and add metadata
        table = pa.Table.from_pylist(records)
        table = table.replace_schema_metadata(metadata)
        
        return table

    def flush(self):
        if self.path and self._rows:
            self._atomic_write(self.path, self._rows)
        if self.events_path and self._events:
            self._atomic_write(self.events_path, self._events)

    def _atomic_write(self, path: str, records: List[Dict[str, Any]]):
        """
        Writes records to a temporary file and then atomically renames it to the final path.
        """
        table = self.to_arrow_table(records)
        
        # Create a temporary file in the same directory to ensure atomic move
        temp_dir = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".tmp") as temp_file:
            temp_path = temp_file.name
            pq.write_table(table, temp_path)
        
        # Atomically rename the temporary file to the final path
        if os.path.exists(path):
            os.remove(path)
        os.rename(temp_path, path)
