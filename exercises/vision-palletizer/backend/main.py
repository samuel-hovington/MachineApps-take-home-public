"""
Vision Palletizer Backend
========================

FastAPI application for coordinating UR5e robot palletizing operations.
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager

from robot.connection import RobotConnection
from api.routes import router as palletizer_router

from robot.motion import MotionController
from palletizer.state_machine import PalletizerStateMachine
from api.routes import set_palletizer


# Global robot connection instance
robot_connection: RobotConnection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - connect/disconnect robot."""
    global robot_connection
    
    # Startup: Initialize robot connection (will auto-reconnect later if needed)
    robot_connection = RobotConnection()
    connected = robot_connection.connect()
    
    if connected:
        print("✓ Robot connection established")
    else:
        print("⚠ Robot not available yet - will auto-reconnect when robot is powered on")
        print("  → Power on the robot in PolyScope, then call /health to connect")
    
    # Initialize motion controller and palletizer
    motion_controller = MotionController(robot_connection)
    palletizer = PalletizerStateMachine(motion_controller)
    set_palletizer(palletizer)  # Make it available to API routes
    print("✓ Palletizer initialized")
    
    yield
    
    # Shutdown: Clean up robot connection
    if robot_connection:
        robot_connection.disconnect()
        print("✓ Robot connection closed")


app = FastAPI(
    title="Vision Palletizer API",
    description="API for controlling UR5e palletizing operations with vision-based picking",
    version="1.0.0",
    lifespan=lifespan,
)


# Include palletizer routes
app.include_router(palletizer_router, prefix="/palletizer", tags=["Palletizer"])


@app.get("/health")
async def health_check():
    """
    Check API and robot connection health.
    
    This endpoint also attempts to reconnect to the robot if disconnected,
    so you can use it to verify when the robot becomes available after
    powering it on in PolyScope.
    """
    robot_status = "disconnected"
    
    if robot_connection:
        # This will attempt to reconnect if not connected
        if robot_connection.check_and_reconnect():
            robot_status = "connected"
    
    return {
        "status": "healthy",
        "robot": robot_status,
    }


@app.get("/")
async def root():
    """API root - redirect to docs."""
    return {
        "message": "Vision Palletizer API",
        "docs": "/docs",
        "health": "/health",
    }


def get_robot_connection() -> RobotConnection | None:
    """Get the global robot connection instance."""
    return robot_connection
