"""
Palletizing Grid Calculations
============================

Calculate place positions for boxes in an N×M grid pattern.
"""

from typing import List, Tuple


def calculate_place_positions(
    rows: int,
    cols: int,
    box_size_mm: Tuple[float, float, float],
    pallet_origin_mm: Tuple[float, float, float],
    spacing_mm: float = 10.0,
) -> List[Tuple[float, float, float]]:
    """
    Calculate TCP positions for placing boxes in a grid pattern.
    
    Args:
        rows: Number of rows (N)
        cols: Number of columns (M)
        box_size_mm: (width, depth, height) of each box in mm
        pallet_origin_mm: (x, y, z) position of the first box placement
        spacing_mm: Gap between adjacent boxes (default 10mm)
    
    Returns:
        List of (x, y, z) TCP target positions, ordered for row-by-row filling.
    """
    positions = []
    
    # Extract dimensions
    width, depth, height = box_size_mm
    origin_x, origin_y, origin_z = pallet_origin_mm
    
    # Calculate positions row-by-row
    for row in range(rows):
        for col in range(cols):
            # X position increases with column (moving right)
            x = origin_x + col * (width + spacing_mm)
            
            # Y position increases with row (moving forward)
            y = origin_y + row * (depth + spacing_mm)
            
            # Z position is at the pallet height (origin Z)
            z = origin_z
            
            positions.append((x, y, z))
    
    return positions
