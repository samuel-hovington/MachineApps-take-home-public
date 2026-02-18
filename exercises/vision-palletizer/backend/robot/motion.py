"""
Motion Controller
================

Implements robot motion commands for pick and place operations.
"""

from typing import Optional
import numpy as np
import csv
import os
from datetime import datetime

from .connection import RobotConnection


class MotionController:
    """
    Controls robot motion for palletizing operations.
    
    Coordinates:
    - All positions are in meters
    - All orientations are in radians (axis-angle representation for UR)
    """
    
    # Safety parameters
    APPROACH_HEIGHT_OFFSET = 0.100  # 100mm above pick/place position
    DEFAULT_VELOCITY = 0.5          # m/s
    DEFAULT_ACCELERATION = 0.5      # m/s²
    
    def __init__(self, connection: RobotConnection):
        """
        Initialize motion controller.
        
        Args:
            connection: Active robot connection instance.
        """
        self.connection = connection
        self._gripper_closed = False
        self._stop_requested = False
        
        # Initialize motion log CSV file
        self.motion_log_file = "motion_log.csv"
        self._init_motion_log()
    
    def _init_motion_log(self) -> None:
        """Initialize the motion log CSV file with headers."""
        file_exists = os.path.exists(self.motion_log_file)
        
        if not file_exists:
            with open(self.motion_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Motion Type', 'Position/Joints', 'Status'])
    
    def _log_motion(self, motion_type: str, data: list, status: str) -> None:
        """
        Log motion execution to CSV file.
        
        Args:
            motion_type: "moveL" or "moveJ"
            data: Position or joint angles as list
            status: "success" or "failed"
        """
        try:
            with open(self.motion_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = datetime.now().isoformat()
                # Format data as comma-separated values in the data column
                data_str = ','.join([f"{x:.6f}" for x in data])
                writer.writerow([timestamp, motion_type, data_str, status])
        except Exception as e:
            print(f"Warning: Failed to log motion to CSV: {e}")
    
    def request_stop(self) -> None:
        """
        Request to stop the current movement.
        Sets the _stop_requested flag to signal movement methods to stop.
        """
        self._stop_requested = True
        print("Stop requested for current movement")
    
    def _reset_stop_flag(self) -> None:
        """
        Reset the stop requested flag at the start of a new movement.
        """
        self._stop_requested = False
    
    
        # Initialize motion log CSV file
        self.motion_log_file = "motion_log.csv"
        self._init_motion_log()
    
    def move_to_home(self) -> bool:
        """
        Move robot to home/safe position.
        
        Returns:
            True if move completed successfully.
        """
        home_pose = [0.5, 0.0, 0.6] + self.get_default_orientation()
        
        return self._move_linear(home_pose, self.DEFAULT_VELOCITY, self.DEFAULT_ACCELERATION)
    
    def move_to_pick(
        self,
        position: list[float],
        orientation: Optional[list[float]] = None,
    ) -> bool:
        """
        Execute pick motion sequence.
        
        Args:
            position: [x, y, z] pick position in robot base frame (meters)
            orientation: [rx, ry, rz] tool orientation (axis-angle, radians)
                        If None, use default downward orientation.
        
        Returns:
            True if pick completed successfully.
        """
        # Use default orientation if not provided
        if orientation is None:
            orientation = self.get_default_orientation()
        
        try:
            # Convert position to numpy array for easier manipulation
            pick_position_m = np.array(position)
            
            # Execute pick sequence:
            # 1. Move to approach position (above the pick target)
            approach_position = pick_position_m.copy()
            approach_position[2] += self.APPROACH_HEIGHT_OFFSET  # Raise Z by approach height
            approach_pose = list(approach_position) + orientation
            
            print(f"Moving to approach position: {approach_pose[:3]}")
            if not self._move_linear(approach_pose, self.DEFAULT_VELOCITY, self.DEFAULT_ACCELERATION):
                return False
            print("Reached approach position")
            
            # 2. Move linearly down to pick position
            pick_pose = list(pick_position_m) + orientation
            print(f"Descending to pick position: {pick_position_m}")
            if not self._move_linear(pick_pose, self.DEFAULT_VELOCITY, self.DEFAULT_ACCELERATION):
                return False
            print("Reached pick position")
            
            # 3. Close gripper
            print("Closing gripper")
            if not self.close_gripper():
                return False
            print("Gripper closed")
            
            # 4. Retract to approach position
            print(f"Retracting to approach position: {approach_pose[:3]}")
            if not self._move_linear(approach_pose, self.DEFAULT_VELOCITY, self.DEFAULT_ACCELERATION):
                return False
            print("Retracted from pick position")
            
            return True
            
        except Exception as e:
            print(f"Error in pick sequence: {e}")
            return False
    
    def move_to_place(
        self,
        position: list[float],
        orientation: Optional[list[float]] = None,
    ) -> bool:
        """
        Execute place motion sequence.
        
        Args:
            position: [x, y, z] place position in robot base frame (meters)
            orientation: [rx, ry, rz] tool orientation (axis-angle, radians)
                        If None, use default downward orientation.
        
        Returns:
            True if place completed successfully.
        """
        # Use default orientation if not provided
        if orientation is None:
            orientation = self.get_default_orientation()
        
        try:
            # Convert position to numpy array for easier manipulation
            place_position_m = np.array(position)
            
            # Execute place sequence:
            # 1. Move to approach position (above the place target)
            approach_position = place_position_m.copy()
            approach_position[2] += self.APPROACH_HEIGHT_OFFSET  # Raise Z by approach height
            approach_pose = list(approach_position) + orientation
            
            print(f"Moving to place approach position: {approach_pose[:3]}")
            if not self._move_linear(approach_pose, self.DEFAULT_VELOCITY, self.DEFAULT_ACCELERATION):
                return False
            print("Reached place approach position")
            
            # 2. Move linearly down to place position
            place_pose = list(place_position_m) + orientation
            print(f"Descending to place position: {place_position_m}")
            if not self._move_linear(place_pose, self.DEFAULT_VELOCITY, self.DEFAULT_ACCELERATION):
                return False
            
            # 3. Open gripper to release object
            print("Opening gripper")
            if not self.open_gripper():
                return False
            print("Gripper opened")
            
            # 4. Retract to approach position
            print(f"Retracting from place position: {approach_pose[:3]}")
            if not self._move_linear(approach_pose, self.DEFAULT_VELOCITY, self.DEFAULT_ACCELERATION):
                return False
            print("Retracted from place position")
            
            return True
            
        except Exception as e:
            print(f"Error in place sequence: {e}")
            return False
    
    def open_gripper(self) -> bool:
        """
        Open the gripper to release object.
        
        Returns:
            True if gripper opened successfully.
        """
        self._gripper_closed = False
        print("[MOCK] Gripper opened")
        return True
    
    def close_gripper(self) -> bool:
        """
        Close the gripper to grasp object.
        
        Returns:
            True if gripper closed successfully.
        """
        self._gripper_closed = True
        print("[MOCK] Gripper closed")
        return True
    
    def _move_linear(
        self,
        pose: list[float],
        velocity: float = DEFAULT_VELOCITY,
        acceleration: float = DEFAULT_ACCELERATION,
    ) -> bool:
        """
        Execute linear move to target pose.
        
        Args:
            pose: [x, y, z, rx, ry, rz] target pose
            velocity: Move velocity in m/s
            acceleration: Move acceleration in m/s²
        
        Returns:
            True if move completed, False if stopped or failed.
        """
        if self.connection.is_mock_mode():
            print(f"[MOCK] moveL to {pose[:3]}")
            self._log_motion("moveL", pose, "success (mock)")
            return True
        
        try:
            if self.connection.control is None:
                print("Error: Robot control interface not available")
                self._log_motion("moveL", pose, "failed (no interface)")
                return False
            
            # Check if stop was requested before starting movement
            if self._stop_requested:
                print("Movement cancelled: stop was requested")
                self._log_motion("moveL", pose, "cancelled (stop requested)")
                self._reset_stop_flag()
                return False
            
            # Execute linear move on real robot
            # asynchronous=False makes the call block until motion completes
            self.connection.control.moveL(pose, velocity, acceleration, asynchronous=False)
            
            # Check if stop was requested after movement completes
            if self._stop_requested:
                print("Movement stopped by stop request")
                self._log_motion("moveL", pose, "stopped")
                return False
            
            self._log_motion("moveL", pose, "success")
            return True
        except Exception as e:
            print(f"Error executing linear move: {e}")
            self._log_motion("moveL", pose, f"failed ({str(e)})")
            return False
    
    def _move_joint(
        self,
        joints: list[float],
        velocity: float = 1.0,
        acceleration: float = 1.0,
    ) -> bool:
        """
        Execute joint move to target configuration.
        
        Args:
            joints: List of 6 joint angles in radians
            velocity: Joint velocity in rad/s
            acceleration: Joint acceleration in rad/s²
        
        Returns:
            True if move completed, False if stopped or failed.
        """
        if self.connection.is_mock_mode():
            print(f"[MOCK] moveJ to {joints}")
            self._log_motion("moveJ", joints, "success (mock)")
            return True
        
        try:
            if self.connection.control is None:
                print("Error: Robot control interface not available")
                self._log_motion("moveJ", joints, "failed (no interface)")
                return False
            
            # Check if stop was requested before starting movement
            if self._stop_requested:
                print("Movement cancelled: stop was requested")
                self._log_motion("moveJ", joints, "cancelled (stop requested)")
                self._reset_stop_flag()
                return False
            
            # Execute joint move on real robot
            # asynchronous=False makes the call block until motion completes
            self.connection.control.moveJ(joints, velocity, acceleration, asynchronous=False)
            
            # Check if stop was requested after movement completes
            if self._stop_requested:
                print("Movement stopped by stop request")
                self._log_motion("moveJ", joints, "stopped")
                return False
            
            self._log_motion("moveJ", joints, "success")
            return True
        except Exception as e:
            print(f"Error executing joint move: {e}")
            self._log_motion("moveJ", joints, f"failed ({str(e)})")
            return False
    
    def get_default_orientation(self) -> list[float]:
        """
        Get default tool orientation for picking (pointing down).
        
        Returns:
            [rx, ry, rz] in axis-angle representation.
        
        Note: For a tool pointing straight down (Z toward floor),
        the rotation from base frame is typically [0, π, 0] or [π, 0, 0]
        depending on your tool frame setup.
        """
        return [0.0, np.pi, 0.0]
