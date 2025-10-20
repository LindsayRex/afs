"""
Telemetry streaming server for real-time dashboard updates.
Provides WebSocket endpoint for broadcasting live telemetry data to dashboard clients.
"""
import asyncio
import json
import logging
import websockets
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)

@dataclass
class TelemetryUpdate:
    """Real-time telemetry update message."""
    run_id: str
    iteration: int
    timestamp: float
    energy: float
    grad_norm: float
    sparsity: float
    phase: str
    certificates: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class TelemetryStreamer:
    """
    WebSocket server for streaming live telemetry data to dashboard clients.
    
    Maintains connections to multiple dashboard clients and broadcasts
    telemetry updates from active optimization runs.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        self._server: Optional[websockets.WebSocketServer] = None
        
    async def start(self):
        """Start the WebSocket server."""
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
        logger.info(f"Telemetry streamer started on ws://{self.host}:{self.port}")
        
    async def stop(self):
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Telemetry streamer stopped")
        
    async def _handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connections."""
        logger.info(f"New dashboard client connected: {websocket.remote_address}")
        self.connected_clients.add(websocket)
        
        try:
            # Send current active runs to new client
            await self._send_active_runs(websocket)
            
            # Keep connection alive and handle client messages
            async for message in websocket:
                await self._handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Dashboard client disconnected: {websocket.remote_address}")
        finally:
            self.connected_clients.discard(websocket)
            
    async def _handle_client_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle messages from dashboard clients."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "subscribe_run":
                run_id = data.get("run_id")
                if run_id:
                    await self._send_run_history(websocket, run_id)
                    
            elif msg_type == "list_runs":
                await self._send_run_list(websocket)
                
            elif msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong", "timestamp": time.time()}))
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message from client: {message}")
            
    async def _send_active_runs(self, websocket: websockets.WebSocketServerProtocol):
        """Send information about currently active runs to a client."""
        active_runs_info = {
            "type": "active_runs",
            "runs": list(self.active_runs.keys()),
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(active_runs_info))
        
    async def _send_run_list(self, websocket: websockets.WebSocketServerProtocol):
        """Send list of all known runs to a client."""
        run_list = {
            "type": "run_list",
            "runs": [
                {
                    "run_id": run_id,
                    "status": info.get("status", "unknown"),
                    "last_update": info.get("last_update", 0),
                    "iteration": info.get("last_iteration", 0)
                }
                for run_id, info in self.active_runs.items()
            ],
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(run_list))
        
    async def _send_run_history(self, websocket: websockets.WebSocketServerProtocol, run_id: str):
        """Send historical telemetry data for a specific run."""
        if run_id in self.active_runs:
            run_info = self.active_runs[run_id]
            history = run_info.get("history", [])
            
            history_message = {
                "type": "run_history",
                "run_id": run_id,
                "history": history,
                "timestamp": time.time()
            }
            await websocket.send(json.dumps(history_message))
            
    async def broadcast_telemetry_update(self, update: TelemetryUpdate):
        """Broadcast a telemetry update to all connected clients."""
        message = {
            "type": "telemetry_update",
            "data": asdict(update),
            "timestamp": time.time()
        }
        
        # Update internal state
        run_id = update.run_id
        if run_id not in self.active_runs:
            self.active_runs[run_id] = {
                "status": "active",
                "start_time": update.timestamp,
                "history": []
            }
            
        run_info = self.active_runs[run_id]
        run_info["last_update"] = update.timestamp
        run_info["last_iteration"] = update.iteration
        run_info["history"].append(asdict(update))
        
        # Keep only recent history (last 1000 points)
        if len(run_info["history"]) > 1000:
            run_info["history"] = run_info["history"][-1000:]
            
        # Broadcast to all clients
        if self.connected_clients:
            message_json = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_json) for client in self.connected_clients],
                return_exceptions=True
            )
            
    def update_run_status(self, run_id: str, status: str):
        """Update the status of a run (active, completed, failed, etc.)."""
        if run_id in self.active_runs:
            self.active_runs[run_id]["status"] = status
            
    def get_active_runs(self) -> List[str]:
        """Get list of currently active run IDs."""
        return list(self.active_runs.keys())
        
    def get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific run."""
        return self.active_runs.get(run_id)

# Global streamer instance
_streamer: Optional[TelemetryStreamer] = None

def get_streamer() -> TelemetryStreamer:
    """Get the global telemetry streamer instance."""
    global _streamer
    if _streamer is None:
        _streamer = TelemetryStreamer()
    return _streamer

async def start_streamer(host: str = "localhost", port: int = 8765):
    """Start the telemetry streaming server."""
    streamer = get_streamer()
    await streamer.start()
    return streamer

def broadcast_telemetry_update(update: TelemetryUpdate):
    """Broadcast a telemetry update (synchronous wrapper)."""
    streamer = get_streamer()
    # Create a task to broadcast asynchronously
    asyncio.create_task(streamer.broadcast_telemetry_update(update))