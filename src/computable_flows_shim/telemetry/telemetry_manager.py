import os
import datetime
from typing import Optional

from .flight_recorder import FlightRecorder
from .manifest_writer import write_manifest

class TelemetryManager:
    """
    Orchestrates the creation and management of telemetry data for a single run.
    """
    def __init__(self, base_path: str, flow_name: str, run_id: Optional[str] = None):
        self.base_path = base_path
        self.flow_name = flow_name
        self.run_id = run_id or self._generate_run_id()
        self.run_path = os.path.join(self.base_path, f"fda_run_{self.run_id}")
        self._flight_recorder: Optional[FlightRecorder] = None
        self._setup_run_directory()

    def _generate_run_id(self) -> str:
        """Generates a run ID in the format YYYYMMDD_HHMMSS."""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _setup_run_directory(self):
        """Creates the directory for the run if it doesn't exist."""
        os.makedirs(self.run_path, exist_ok=True)

    @property
    def flight_recorder(self) -> FlightRecorder:
        """
        Returns a FlightRecorder instance for the current run.
        Initializes the FlightRecorder if it hasn't been already.
        """
        if self._flight_recorder is None:
            telemetry_path = os.path.join(self.run_path, "telemetry.parquet")
            events_path = os.path.join(self.run_path, "events.parquet")
            self._flight_recorder = FlightRecorder(path=telemetry_path, events_path=events_path)
        return self._flight_recorder

    def write_run_manifest(self, schema_version: int, residual_details: dict, extra: Optional[dict] = None):
        """
        Writes the manifest.toml file for the current run.
        """
        write_manifest(
            out_dir=self.run_path,
            schema_version=schema_version,
            flow_name=self.flow_name,
            run_id=self.run_id,
            residual_details=residual_details,
            extra=extra,
        )

    def flush(self):
        """
        Flushes any buffered telemetry data to disk.
        """
        if self._flight_recorder:
            self._flight_recorder.flush()

