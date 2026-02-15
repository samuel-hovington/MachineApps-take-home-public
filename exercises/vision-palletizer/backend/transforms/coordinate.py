"""
Coordinate Transformations
=========================

Transform coordinates between camera frame and robot base frame.

Refer to the README for camera mounting specifications.
"""

import numpy as np

CAMERA_POSITION = np.array([500, 300, 0])  # mm
CAMERA_ORIENTATION_DEG = (15, -10, 45)  # Roll, Pitch, Yaw in degrees


def build_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from Roll-Pitch-Yaw (Euler) angles.
    
    Args:
        roll: Rotation about X-axis in radians
        pitch: Rotation about Y-axis in radians
        yaw: Rotation about Z-axis in radians
    
    Returns:
        3x3 rotation matrix
    """
    # Rotation about X-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Rotation about Y-axis (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation about Z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: Rz * Ry * Rx
    return Rz @ Ry @ Rx


def camera_to_robot(point_camera: np.ndarray) -> np.ndarray:
    """
    Transform a point from camera frame to robot base frame.
    
    Args:
        point_camera: [x, y, z] coordinates in camera frame (mm)
    
    Returns:
        [x, y, z] coordinates in robot base frame (mm)
    """
    # Camera mounting parameters from README
    camera_position = CAMERA_POSITION  # mm
    roll_deg, pitch_deg, yaw_deg = CAMERA_ORIENTATION_DEG
    
    # Convert degrees to radians
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    
    # Build rotation matrix from camera to robot frame
    rotation = build_rotation_matrix(roll, pitch, yaw)
    
    # Apply transformation: rotate then translate
    point_robot = rotation @ point_camera + camera_position
    
    return point_robot


def robot_to_camera(point_robot: np.ndarray) -> np.ndarray:
    """
    Transform a point from robot base frame to camera frame.
    
    Args:
        point_robot: [x, y, z] coordinates in robot base frame (mm)
    
    Returns:
        [x, y, z] coordinates in camera frame (mm)
    """
    # Camera mounting parameters from README
    camera_position = np.array([500, 300, 800])  # mm
    roll_deg, pitch_deg, yaw_deg = 15, -10, 45
    
    # Convert degrees to radians
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    
    # Build rotation matrix from camera to robot frame
    rotation = build_rotation_matrix(roll, pitch, yaw)
    
    # Apply inverse transformation: subtract translation then apply inverse rotation
    # Since rotation matrix is orthogonal, inverse = transpose
    point_camera = rotation.T @ (point_robot - camera_position)
    
    return point_camera


def build_homogeneous_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """
    Build a 4x4 homogeneous transformation matrix.
    
    Args:
        rotation: 3x3 rotation matrix
        translation: 3x1 or (3,) translation vector
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    # Ensure translation is a 1D array for proper reshape
    translation = np.asarray(translation).flatten()
    
    # Create 4x4 identity matrix
    transform = np.eye(4)
    
    # Insert rotation matrix in top-left 3x3 block
    transform[:3, :3] = rotation
    
    # Insert translation vector in top-right column
    transform[:3, 3] = translation
    
    return transform
