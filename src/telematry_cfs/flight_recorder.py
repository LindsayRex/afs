import pandas as pd
from typing import Any, Dict, List

class FlightRecorder:
    def __init__(self, path: str = None):
        self.path = path
        self._rows: List[Dict[str, Any]] = []
    def log(self, **kwargs):
        self._rows.append(dict(kwargs))
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)
    # Placeholder for future Parquet writing
    def flush(self):
        if self.path:
            df = self.to_dataframe()
            df.to_parquet(self.path, index=False)
