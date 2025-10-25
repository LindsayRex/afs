"""
Checkpointing system for Computable Flows optimization flows.
Provides atomic saves, resume, and rollback capabilities for long-running optimizations.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp


class CheckpointManager:
    """
    Manages checkpoints for AFS optimization flows.

    Provides atomic saves, resume, and rollback capabilities.
    Checkpoints include:
    - Current optimization state (parameters)
    - Iteration count and metadata
    - Telemetry history
    - Flow configuration
    - Certificate values
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def create_checkpoint(
        self,
        run_id: str,
        iteration: int,
        state: dict[str, jnp.ndarray],
        flow_config: dict[str, Any],
        telemetry_history: list[dict[str, Any]] | None = None,
        certificates: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create an atomic checkpoint of the current optimization state.

        Args:
            run_id: Unique identifier for the run
            iteration: Current iteration number
            state: Current optimization state (parameters)
            flow_config: Flow configuration
            telemetry_history: Optional telemetry data history
            certificates: Optional certificate values (eta_dd, gamma)
            metadata: Optional additional metadata

        Returns:
            Checkpoint filename
        """
        timestamp = datetime.now().isoformat()
        # Sanitize timestamp for filename (replace colons and dots)
        safe_timestamp = timestamp.replace(":", "").replace(".", "_")
        checkpoint_id = f"{run_id}_iter_{iteration}_{safe_timestamp}"

        # Create checkpoint data structure
        checkpoint_data = {
            "run_id": run_id,
            "iteration": iteration,
            "timestamp": timestamp,
            "checkpoint_id": checkpoint_id,
            "flow_config": flow_config,
            "certificates": certificates or {},
            "metadata": metadata or {},
            "telemetry_summary": self._summarize_telemetry(telemetry_history)
            if telemetry_history
            else {},
        }

        # Convert JAX arrays to serializable format
        serializable_state = {}
        for key, value in state.items():
            if isinstance(value, jnp.ndarray):
                serializable_state[key] = {
                    "data": value.tolist(),
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                }
            else:
                serializable_state[key] = value

        checkpoint_data["state"] = serializable_state

        # Save telemetry history separately if provided
        if telemetry_history:
            telemetry_file = self.checkpoint_dir / f"{checkpoint_id}_telemetry.pkl"
            with open(telemetry_file, "wb") as f:
                pickle.dump(telemetry_history, f)
            checkpoint_data["telemetry_file"] = str(telemetry_file)

        # Atomic save: write to temp file first, then rename
        temp_file = self.checkpoint_dir / f"{checkpoint_id}.tmp"
        final_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        with open(temp_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        # Atomic rename
        temp_file.rename(final_file)

        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        """
        Load a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Dictionary containing checkpoint data with reconstructed JAX arrays
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")

        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)

        # Reconstruct JAX arrays
        reconstructed_state = {}
        for key, value in checkpoint_data["state"].items():
            if isinstance(value, dict) and "data" in value:
                # Reconstruct JAX array
                array_data = jnp.array(value["data"], dtype=value["dtype"])
                reconstructed_state[key] = array_data.reshape(value["shape"])
            else:
                reconstructed_state[key] = value

        checkpoint_data["state"] = reconstructed_state

        # Load telemetry history if available
        if "telemetry_file" in checkpoint_data:
            telemetry_file = Path(checkpoint_data["telemetry_file"])
            if telemetry_file.exists():
                with open(telemetry_file, "rb") as f:
                    checkpoint_data["telemetry_history"] = pickle.load(f)
            else:
                checkpoint_data["telemetry_history"] = []

        return checkpoint_data

    def list_checkpoints(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """
        List available checkpoints, optionally filtered by run_id.

        Args:
            run_id: Optional run ID to filter by

        Returns:
            List of checkpoint metadata
        """
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)

                if run_id is None or data.get("run_id") == run_id:
                    # Add file info
                    data["file_path"] = str(checkpoint_file)
                    data["file_size"] = checkpoint_file.stat().st_size
                    checkpoints.append(data)

            except (json.JSONDecodeError, KeyError):
                # Skip corrupted checkpoint files
                continue

        # Sort by timestamp (most recent first)
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return checkpoints

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint and its associated files.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if successfully deleted
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        telemetry_file = self.checkpoint_dir / f"{checkpoint_id}_telemetry.pkl"

        deleted = False

        if checkpoint_file.exists():
            checkpoint_file.unlink()
            deleted = True

        if telemetry_file.exists():
            telemetry_file.unlink()

        return deleted

    def get_latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """
        Get the most recent checkpoint for a run.

        Args:
            run_id: Run identifier

        Returns:
            Latest checkpoint data or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints(run_id)
        return checkpoints[0] if checkpoints else None

    def rollback_to_checkpoint(
        self, checkpoint_id: str, target_run_id: str
    ) -> dict[str, Any]:
        """
        Create a new checkpoint by rolling back to a previous state.
        Useful for experimentation or recovery from failed optimization paths.

        Args:
            checkpoint_id: Source checkpoint to rollback from
            target_run_id: New run ID for the rolled-back checkpoint

        Returns:
            New checkpoint data
        """
        source_checkpoint = self.load_checkpoint(checkpoint_id)

        # Create new checkpoint with updated metadata
        new_metadata = source_checkpoint.get("metadata", {}).copy()
        new_metadata["rollback_from"] = checkpoint_id
        new_metadata["rollback_timestamp"] = datetime.now().isoformat()

        new_checkpoint_id = self.create_checkpoint(
            run_id=target_run_id,
            iteration=source_checkpoint["iteration"],
            state=source_checkpoint["state"],
            flow_config=source_checkpoint["flow_config"],
            telemetry_history=source_checkpoint.get("telemetry_history"),
            certificates=source_checkpoint.get("certificates"),
            metadata=new_metadata,
        )

        return self.load_checkpoint(new_checkpoint_id)

    def _summarize_telemetry(
        self, telemetry_history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Create a summary of telemetry history for quick checkpoint loading.

        Args:
            telemetry_history: Full telemetry history

        Returns:
            Summary statistics
        """
        if not telemetry_history:
            return {}

        summary = {
            "total_iterations": len(telemetry_history),
            "final_energy": telemetry_history[-1].get("E", 0),
            "final_grad_norm": telemetry_history[-1].get("grad_norm", 0),
            "final_sparsity": telemetry_history[-1].get("sparsity_wx", 0),
            "energy_range": {
                "min": min(t.get("E", float("inf")) for t in telemetry_history),
                "max": max(t.get("E", float("-inf")) for t in telemetry_history),
            },
            "phase_distribution": {},
        }

        # Count phases
        phases = {}
        for t in telemetry_history:
            phase = t.get("phase", "UNKNOWN")
            phases[phase] = phases.get(phase, 0) + 1

        summary["phase_distribution"] = phases

        return summary

    def cleanup_old_checkpoints(self, run_id: str, keep_last: int = 5) -> int:
        """
        Clean up old checkpoints for a run, keeping only the most recent ones.

        Args:
            run_id: Run identifier
            keep_last: Number of most recent checkpoints to keep

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints(run_id)

        if len(checkpoints) <= keep_last:
            return 0

        # Delete older checkpoints
        deleted_count = 0
        for checkpoint in checkpoints[keep_last:]:
            if self.delete_checkpoint(checkpoint["checkpoint_id"]):
                deleted_count += 1

        return deleted_count
