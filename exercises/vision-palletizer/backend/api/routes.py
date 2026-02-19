"""
Palletizer API Routes
====================

FastAPI routes for palletizer control.
"""

import json
import os
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import threading

from palletizer.state_machine import PalletizerStateMachine
from palletizer.grid import calculate_place_positions
from transforms.coordinate import camera_to_robot, camera_yaw_to_robot_yaw

router = APIRouter()

# Global state machine instance (initialized in main.py)
palletizer: Optional[PalletizerStateMachine] = None


def set_palletizer(state_machine: PalletizerStateMachine):
    """Set the global palletizer state machine instance."""
    global palletizer
    palletizer = state_machine


# ============================================================================
# Request/Response Models
# ============================================================================

class PalletConfig(BaseModel):
    """Configuration for palletizing operation."""
    
    rows: int = Field(..., ge=1, le=10, description="Number of rows in the grid")
    cols: int = Field(..., ge=1, le=10, description="Number of columns in the grid")
    box_width_mm: float = Field(..., gt=0, description="Box width in mm (X direction)")
    box_depth_mm: float = Field(..., gt=0, description="Box depth in mm (Y direction)")
    box_height_mm: float = Field(..., gt=0, description="Box height in mm (Z direction)")
    pallet_origin_x_mm: float = Field(..., description="Pallet origin X in mm")
    pallet_origin_y_mm: float = Field(..., description="Pallet origin Y in mm")
    pallet_origin_z_mm: float = Field(..., description="Pallet origin Z in mm")
    
    class Config:
        json_schema_extra = {
            "example": {
                "rows": 2,
                "cols": 2,
                "box_width_mm": 100.0,
                "box_depth_mm": 100.0,
                "box_height_mm": 50.0,
                "pallet_origin_x_mm": 400.0,
                "pallet_origin_y_mm": -200.0,
                "pallet_origin_z_mm": 100.0,
            }
        }


class VisionDetection(BaseModel):
    """Simulated vision detection of a box."""
    
    x_mm: float = Field(..., description="Box X position in camera frame (mm)")
    y_mm: float = Field(..., description="Box Y position in camera frame (mm)")
    z_mm: float = Field(..., description="Box Z position in camera frame (mm)")
    yaw_deg: Optional[float] = Field(0.0, description="Box rotation about Z (degrees)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "x_mm": 50.0,
                "y_mm": -30.0,
                "z_mm": 0.0,
                "yaw_deg": 15.0,
            }
        }


class StatusResponse(BaseModel):
    """Palletizer status response."""
    
    state: str = Field(..., description="Current state machine state")
    current_box: int = Field(..., description="Current box index (0-based)")
    total_boxes: int = Field(..., description="Total boxes to palletize")
    error: Optional[str] = Field(None, description="Error message if in FAULT state")


class ConfigResponse(BaseModel):
    """Configuration response."""
    
    success: bool
    message: str
    grid_size: Optional[str] = None


class CommandResponse(BaseModel):
    """Generic command response."""
    
    success: bool
    message: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/configure", response_model=ConfigResponse)
async def configure_palletizer(config: PalletConfig):
    """
    Configure the palletizing operation.
    
    Sets up the grid dimensions, box size, and pallet origin.
    Can only be called when the palletizer is in IDLE state.
    """
    if palletizer is None:
        raise HTTPException(status_code=503, detail="Palletizer not initialized")
    
    # Prepare parameters
    box_size = (config.box_width_mm, config.box_depth_mm, config.box_height_mm)
    pallet_origin = (config.pallet_origin_x_mm, config.pallet_origin_y_mm, config.pallet_origin_z_mm)
    
    # Configure the state machine
    if not palletizer.configure(config.rows, config.cols, box_size, pallet_origin):
        raise HTTPException(status_code=400, detail="Cannot configure - palletizer not in IDLE state")
    
    # Calculate place positions
    try:
        positions = calculate_place_positions(config.rows, config.cols, box_size, pallet_origin)
        palletizer.context.place_positions = positions
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to calculate positions: {str(e)}")
    
    return ConfigResponse(
        success=True,
        message=f"Configured {config.rows}x{config.cols} grid With box size {box_size} mm at origin {pallet_origin} mm",
        grid_size=f"{config.rows}x{config.cols}"
    )


@router.post("/start", response_model=CommandResponse)
async def start_palletizer():
    """
    Start the palletizing sequence.
    
    Begins the pick-and-place cycle. The palletizer must be configured first.
    """
    if palletizer is None:
        raise HTTPException(status_code=503, detail="Palletizer not initialized")
    
    # Verify configuration is complete
    if palletizer.context.total_boxes == 0:
        raise HTTPException(status_code=400, detail="Palletizer not configured - call /configure first")
    
    if hasattr(palletizer, "_worker_thread") and palletizer._worker_thread.is_alive():
        raise HTTPException(status_code=400, detail="Already running")
    
    # Start the sequence in a thread so this API call returns immediately
    def worker():
        palletizer.begin()  # will block in this thread
    threading.Thread(target=worker, daemon=True).start()
    
    return CommandResponse(
        success=True,
        message=f"Started palletizing sequence - {palletizer.context.total_boxes} boxes to place"
    )


@router.post("/stop", response_model=CommandResponse)
async def stop_palletizer():
    """
    Stop the palletizing sequence.
    
    Gracefully stops the operation and returns to IDLE state.
    """
    if palletizer is None:
        raise HTTPException(status_code=503, detail="Palletizer not initialized")

    # Request stop on the state machine
    if not palletizer.stop():
        raise HTTPException(status_code=400, detail="Failed to stop palletizer")
    

    return CommandResponse(success=True, message="Stop requested; palletizer returning to IDLE")


@router.post("/reset", response_model=CommandResponse)
async def reset_palletizer():
    """
    Reset state to IDLE.
    
    Clears the fault and returns to IDLE state.
    """
    if palletizer is None:
        raise HTTPException(status_code=503, detail="Palletizer not initialized")
    
    if not palletizer.reset():
        raise HTTPException(status_code=400, detail="Reset failed, please try again")
    
    return CommandResponse(
        success=True,
        message="Palletizer reset to IDLE"
    )


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get current palletizer status.
    
    Returns the current state, progress, and any error messages.
    """
    if palletizer is None:
        raise HTTPException(status_code=503, detail="Palletizer not initialized")
    
    # Get progress from state machine
    progress = palletizer.progress
    
    return StatusResponse(
        state=progress["state"],
        current_box=progress["current_box"],
        total_boxes=progress["total_boxes"],
        error=progress["error"]
    )


@router.post("/vision/detect", response_model=CommandResponse)
async def simulate_vision_detection(detection: VisionDetection):
    """
    Simulate a vision detection event.
    
    In a real system, this would come from the vision system.
    For this exercise, use this endpoint to simulate box detections.
    
    The coordinates are in the camera frame and must be transformed
    to the robot frame before use.
    """
    if palletizer is None:
        raise HTTPException(status_code=503, detail="Palletizer not initialized")
    yaw = camera_yaw_to_robot_yaw(detection.yaw_deg)
    detection_in_camera = np.array([detection.x_mm, detection.y_mm, detection.z_mm])
    detection_in_robot = camera_to_robot(detection_in_camera)
    palletizer.context.pick_positions.append((float(detection_in_robot[0]), float(detection_in_robot[1]), float(detection_in_robot[2]), yaw))
    
    return CommandResponse(
        success=True,
        message=f"Vision detection received: ({detection.x_mm:.1f}, {detection.y_mm:.1f}, {detection.z_mm:.1f}) mm in camera frame"
    )


@router.get("/vision/load-detections")
async def load_detections_from_file():
    """
    Load all detections from the camera_detections.json file.
    
    Returns a list of all detected boxes for testing/debugging.
    """
    try:
        # Get the path to the data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, "..", "data", "camera_detections.json")
        
        if not os.path.exists(data_file):
            raise HTTPException(status_code=404, detail=f"Camera detections file not found at {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)

        detections_list = data.get("detections", [])
        
        # Transform each detection individually from camera to robot frame
        detections_in_camera = []
        detections_in_robot = []
        
        for d in detections_list:
            camera_pos = np.array([d["x_mm"], d["y_mm"], d["z_mm"]], dtype=float)
            robot_pos = camera_to_robot(camera_pos)
            robot_yaw = camera_yaw_to_robot_yaw(d.get("yaw_deg", 0.0))
            
            # Store as plain lists for JSON serialization
            detections_in_camera.append([float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])])
            detections_in_robot.append((float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2]), robot_yaw))
        
        # Store the transformed positions in palletizer context
        if palletizer is not None:
            palletizer.context.pick_positions = detections_in_robot
        
        return {
            "success": True,
            "count": len(detections_in_camera),
            "detections_camera_frame": detections_in_camera,
            "detections_robot_frame": [[float(pos[0]), float(pos[1]), float(pos[2]), float(pos[3])] for pos in detections_in_robot]
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in camera detections file: {str(e)}")
    except TypeError as e:
        raise HTTPException(status_code=500, detail=f"Type error processing detections: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading detections: {str(e)}")

# ============================================================================
# Helper/Debug Endpoints (Optional)
# ============================================================================

@router.get("/debug/positions")
async def get_calculated_positions():
    """
    Debug endpoint: Get all calculated place positions.
    
    Useful for verifying grid calculations without running the full sequence.
    """
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post("/debug/transform")
async def test_transform(detection: VisionDetection):
    """
    Debug endpoint: Test coordinate transformation.
    
    Transforms the input coordinates and returns both camera and robot frame values.
    Useful for verifying transformation math.
    """
    raise HTTPException(status_code=501, detail="Not implemented")
