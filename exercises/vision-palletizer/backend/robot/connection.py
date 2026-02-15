"""
Robot Connection Helper
======================

This module handles the connection to URSim via RTDE.
This file is PROVIDED - you should not need to modify it.

Features:
- Auto-reconnection: Automatically attempts to reconnect when robot becomes available
- Lazy connection: Connects on first use, not on initialization
"""

import os
import time
from typing import Optional

try:
    import rtde_control
    import rtde_receive
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    print("Warning: ur_rtde not available. Running in mock mode.")


class RobotConnection:
    """
    Manages connection to Universal Robot via RTDE interface.
    
    Provides access to:
    - rtde_control: For sending motion commands
    - rtde_receive: For reading robot state/telemetry
    
    The connection is lazy and will auto-reconnect when the robot
    becomes available (e.g., after powering on in PolyScope).
    """
    
    # Minimum time between reconnection attempts (seconds)
    RECONNECT_INTERVAL = 2.0
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 30004,
    ):
        """
        Initialize robot connection.
        
        Args:
            host: Robot IP address (default: from ROBOT_HOST env var or 'localhost')
            port: RTDE port (default: 30004)
        """
        self.host = host or os.getenv("ROBOT_HOST", "localhost")
        self.port = port
        
        self._rtde_c: Optional["rtde_control.RTDEControlInterface"] = None
        self._rtde_r: Optional["rtde_receive.RTDEReceiveInterface"] = None
        self._connected = False
        self._mock_mode = not RTDE_AVAILABLE
        self._last_connect_attempt = 0.0
    
    def connect(self) -> bool:
        """
        Establish connection to the robot.
        
        Returns:
            True if connection successful, False otherwise.
        """
        if self._mock_mode:
            print(f"[MOCK] Would connect to robot at {self.host}")
            self._connected = True
            return True
        
        # Rate limit connection attempts
        now = time.time()
        if now - self._last_connect_attempt < self.RECONNECT_INTERVAL:
            return False
        self._last_connect_attempt = now
        
        # Clean up any existing connections first
        self._cleanup_connections()
        
        try:
            print(f"Connecting to robot at {self.host}...")
            self._rtde_c = rtde_control.RTDEControlInterface(self.host)
            print("✓ Control interface connected")
            self._rtde_r = rtde_receive.RTDEReceiveInterface(self.host)
            self._connected = True
            print(f"✓ Connected to robot at {self.host}")
            return True
        except Exception as e:
            print(f"⚠ Failed to connect to robot: {e}")
            self._connected = False
            self._rtde_c = None
            self._rtde_r = None
            return False
    
    def _cleanup_connections(self) -> None:
        """Clean up existing RTDE connections."""
        try:
            if self._rtde_c:
                self._rtde_c.stopScript()
                self._rtde_c.disconnect()
        except Exception:
            pass
        try:
            if self._rtde_r:
                self._rtde_r.disconnect()
        except Exception:
            pass
        self._rtde_c = None
        self._rtde_r = None
        self._connected = False
    
    def disconnect(self) -> None:
        """Disconnect from the robot."""
        if self._mock_mode:
            print("[MOCK] Disconnected from robot")
            self._connected = False
            return
        
        self._cleanup_connections()
        print("Disconnected from robot")
    
    def ensure_connected(self) -> bool:
        """
        Ensure robot is connected, attempting reconnection if needed.
        
        Returns:
            True if connected (or reconnected successfully), False otherwise.
        """
        if self.is_connected():
            return True
        
        # Try to reconnect
        return self.connect()
    
    def is_connected(self) -> bool:
        """
        Check if robot is connected.
        
        This performs an actual connection check and will detect
        if the robot has been disconnected (e.g., powered off).
        """
        if self._mock_mode:
            return self._connected
        
        if not self._connected or self._rtde_c is None or self._rtde_r is None:
            return False
        
        try:
            # Actually check if the connection is still alive
            return self._rtde_c.isConnected()
        except Exception:
            # Connection check failed, mark as disconnected
            self._connected = False
            return False
    
    def check_and_reconnect(self) -> bool:
        """
        Check connection status and attempt reconnection if disconnected.
        
        This is useful to call periodically (e.g., in health checks)
        to automatically reconnect when the robot becomes available.
        
        Returns:
            True if connected, False otherwise.
        """
        if self.is_connected():
            return True
        
        # Not connected, try to reconnect
        return self.connect()
    
    @property
    def control(self) -> Optional["rtde_control.RTDEControlInterface"]:
        """Get RTDE control interface for sending commands."""
        self.ensure_connected()
        return self._rtde_c
    
    @property
    def receive(self) -> Optional["rtde_receive.RTDEReceiveInterface"]:
        """Get RTDE receive interface for reading telemetry."""
        self.ensure_connected()
        return self._rtde_r
    
    def get_tcp_pose(self) -> list[float]:
        """
        Get current TCP (Tool Center Point) pose.
        
        Returns:
            List of [x, y, z, rx, ry, rz] in meters and radians.
        """
        if self._mock_mode:
            # Return a mock home position
            return [0.0, -0.4, 0.4, 0.0, 3.14159, 0.0]
        
        if not self.ensure_connected() or self._rtde_r is None:
            raise RuntimeError("Robot not connected")
        
        return self._rtde_r.getActualTCPPose()
    
    def get_joint_positions(self) -> list[float]:
        """
        Get current joint positions.
        
        Returns:
            List of 6 joint angles in radians.
        """
        if self._mock_mode:
            return [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        
        if not self.ensure_connected() or self._rtde_r is None:
            raise RuntimeError("Robot not connected")
        
        return self._rtde_r.getActualQ()
    
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode (no real robot connection)."""
        return self._mock_mode
