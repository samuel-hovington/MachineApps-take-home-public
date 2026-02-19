"""
Palletizer State Machine

Manages the lifecycle of palletizing operations using vention-state-machine.
Documentation: https://docs.vention.io/docs/state-machine
"""

from enum import Enum, auto
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import threading

from state_machine.core import StateMachine, BaseTriggers
from state_machine.defs import StateGroup, State, Trigger
from state_machine.decorators import on_enter_state, on_state_change

class PositionLogger:
    """Utility class for logging every position sent to the robot. At the end write it to a csv file."""
    def __init__(self, filename: str = "position_log.csv"):
        self.filename = filename
        self.positions = []
    
    def log_position(self, position: list[float]):
        """Log a position (list of floats) to the internal list."""
        self.positions.append(position)
    
    def save_to_csv(self):
        """Save all logged positions to a CSV file."""
        import csv
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "z"])  # Header
            writer.writerows(self.positions)
        print(f"✓ Saved {len(self.positions)} positions to {self.filename}")


class PalletizerState(Enum):
    """Palletizer operation states."""
    IDLE = auto()
    HOMING = auto()
    PICKING = auto()
    PLACING = auto()
    FAULT = auto()


class Running(StateGroup):
    """Active operation states."""
    homing: State = State()
    picking: State = State()
    placing: State = State()


class States:
    running = Running()


class Triggers:
    """Named events that initiate transitions."""
    finished_homing = Trigger("finished_homing")
    finished_picking = Trigger("finished_picking")
    finished_placing = Trigger("finished_placing")
    cycle_complete = Trigger("cycle_complete")
    stop = Trigger("stop")
    reset = Trigger("reset")
    waiting_for_detections = Trigger("waiting_for_detections")


TRANSITIONS = [
    Trigger("start").transition("ready", States.running.homing),
    Triggers.finished_homing.transition(States.running.homing, States.running.picking),
    Triggers.finished_picking.transition(States.running.picking, States.running.placing),
    Triggers.finished_placing.transition(States.running.placing, States.running.picking),
    Triggers.cycle_complete.transition(States.running.placing, "ready"),
    Triggers.waiting_for_detections.transition(States.running.picking, States.running.picking),
    Triggers.stop.transition(States.running.homing, "ready"),
    Triggers.stop.transition(States.running.picking, "ready"),
    Triggers.stop.transition(States.running.placing, "ready"),
    Triggers.reset.transition(States.running.homing, "ready"),
    Triggers.reset.transition(States.running.picking, "ready"),
    Triggers.reset.transition(States.running.placing, "ready"),
    Triggers.reset.transition("ready", "ready"),  # Allow reset from IDLE to IDLE to clear errors
]


@dataclass
class PalletizerContext:
    """Shared context for state machine operations."""
    rows: int = 2
    cols: int = 2
    box_size_mm: tuple[float, float, float] = (100.0, 100.0, 50.0)
    pallet_origin_mm: tuple[float, float, float] = (400.0, -200.0, 100.0)
    current_box_index: int = 0
    total_boxes: int = 0
    pick_position: Optional[tuple[float, float, float, float]] = None
    pick_positions: list[tuple[float, float, float, float]] = field(default_factory=list)
    place_positions: list[tuple[float, float, float, float]] = field(default_factory=list)
    error_message: str = ""


class PalletizerStateMachine(StateMachine):
    """
    State machine for palletizing operations.
    
    Usage:
        machine = PalletizerStateMachine()
        machine.trigger('start')  # Transitions to HOMING
    """
    
    def __init__(self, motion_controller=None):
        super().__init__(
            states=States,
            transitions=TRANSITIONS,
            enable_last_state_recovery=False,
        )
        self.context = PalletizerContext()
        self.motion_controller = motion_controller
        # Use threading.Event for thread-safe stop signaling
        self._stop_requested = threading.Event()
    
    @property
    def current_state(self) -> PalletizerState:
        """Get current state. Note: library uses format 'Running_homing' not 'running.homing'."""
        state_str = self.state
        mapping = {
            "ready": PalletizerState.IDLE,
            "fault": PalletizerState.FAULT,
            "Running_homing": PalletizerState.HOMING,
            "Running_picking": PalletizerState.PICKING,
            "Running_placing": PalletizerState.PLACING,
        }
        return mapping.get(state_str, PalletizerState.IDLE)
    
    @property
    def progress(self) -> dict:
        """Get current progress: state, current_box, total_boxes, error."""
        return {
            "state": self.current_state.name,
            "current_box": self.context.current_box_index,
            "total_boxes": self.context.total_boxes,
            "error": self.context.error_message if self.context.error_message else None,
        }
    
    def configure(
        self,
        rows: int,
        cols: int,
        box_size_mm: tuple[float, float, float],
        pallet_origin_mm: tuple[float, float, float],
    ) -> bool:
        """Configure palletizing parameters. Only valid in IDLE state."""
        if self.current_state != PalletizerState.IDLE:
            return False
        
        self.context.rows = rows
        self.context.cols = cols
        self.context.box_size_mm = box_size_mm
        self.context.pallet_origin_mm = pallet_origin_mm
        self.context.total_boxes = rows * cols
        self.context.current_box_index = 0
        self.context.place_positions = []
        return True
    
    def begin(self) -> bool:
        """Start the palletizing sequence."""
        if self.current_state != PalletizerState.IDLE:
            return False
        try:
            self.trigger("start")
            return True
        except Exception:
            return False
    
    def stop(self) -> bool:
        """Stop the palletizing sequence and return to IDLE."""
        if self.current_state == PalletizerState.IDLE:
            # If already idle, ensure any previous stop flags are cleared
            self._stop_requested.clear()
            try:
                # reset controller stop flag if available
                if self.motion_controller:
                    self.motion_controller._reset_stop_flag()
            except Exception:
                pass
            return True
        # Set flag to request stop - this will be checked by state handlers
        self._stop_requested.set()
        # Also request stop on the motion controller (if present)
        try:
            if self.motion_controller:
                self.motion_controller.request_stop()
        except Exception:
            pass
        try:
            self.trigger("stop")
            return True
        except Exception:
            return False
    
    def reset(self) -> bool:
        """Reset from FAULT state to IDLE."""
        try:
            self.motion_controller._reset_stop_flag()  # Clear any stop requests before resetting
            if not self.motion_controller.move_to_home():
                print("Failed to move to home position during reset")
                # Motion failed, transition to FAULT state
                self.fault("Failed to move to home position")
            print("Palletizer reset to IDLE state")
            self.trigger("reset")
            print("State machine reset to IDLE")
            self.context.error_message = ""
            self.context.current_box_index = 0
            
            return True
        except Exception:
            return False
    
    def fault(self, message: str) -> bool:
        """Transition to FAULT state with an error message."""
        self.context.error_message = message
        try:
            self.trigger(BaseTriggers.TO_FAULT.value)
            return True
        except Exception:
            return False

    # State Entry Callbacks - Implement your business logic here
    
    @on_enter_state(States.running.homing)
    def on_enter_homing(self, _):
        """
        Execute homing sequence:
        1. Command robot to move to home position
        2. Wait for motion to complete
        3. Call self.trigger('finished_homing') when done
        """
        try:
            # Check if stop has been requested
            if self._stop_requested.is_set():
                self._stop_requested.clear()
                print("Homing operation cancelled due to stop request")
                return
            
            # Execute home motion
            if self.motion_controller.move_to_home():
                # Trigger transition to PICKING state
                self.trigger('finished_homing')
            else:
                # Motion failed, transition to FAULT state
                self.fault("Failed to move to home position")
        except Exception as e:
            self.fault(f"Homing error: {str(e)}")
    
    @on_enter_state(States.running.picking)
    def on_enter_picking(self, _):
        """
        Execute pick sequence:
        1. Get next pick position from camera detections (already transformed to robot frame)
        2. Execute pick motion via motion controller (approach -> descend -> grip -> retract)
        3. Trigger 'finished_picking' when done
        """
        try:
            # Check if stop has been requested
            if self._stop_requested.is_set():
                self._stop_requested.clear()
                print("Pick operation cancelled due to stop request")
                return
            
            # Check if we have pick positions available
            if not self.context.pick_positions:
                self.fault("No pick positions available - load detections first with /vision/load-detections")
                return
            
            # Check if we have more boxes to pick
            if self.context.current_box_index >= self.context.total_boxes:
                self.fault("Box index out of range - all boxes should have been picked")
                return
            
            if self.context.current_box_index >= len(self.context.pick_positions):
                self.trigger("waiting_for_detections")  # Stay in picking state and wait for more detections
                return
            
            # Get the next pick position (already in robot frame, in mm)
            self.context.pick_position = self.context.pick_positions[self.context.current_box_index]
            print(f"Picking box {self.context.current_box_index + 1}: position (robot frame) {self.context.pick_position}")
            
            # Convert from mm to meters for motion controller
            pick_position_m = np.array(self.context.pick_position[:-1]) / 1000.0
            yaw_rad = np.radians(self.context.pick_position[-1])  # Extract yaw from the last element
            
            #compute the orientation for picking based on the yaw (assuming we want to keep the gripper pointing down)
            pick_orientation = [np.pi * np.cos(yaw_rad), np.pi * np.sin(yaw_rad), 0.0]
            
            # Execute pick sequence using motion controller
            if not self.motion_controller.move_to_pick(list(pick_position_m), pick_orientation):
                self.fault("Pick motion failed")
                return
            
            # Pick sequence completed successfully
            print(f"Successfully picked box {self.context.current_box_index + 1}")
            self.trigger('finished_picking')
            
        except Exception as e:
            self.fault(f"Pick error: {str(e)}")
    
    @on_enter_state(States.running.placing)
    def on_enter_placing(self, _):
        """
        Execute place sequence:
        1. Get next place position from calculated grid
        2. Execute place motion via motion controller (approach -> descend -> release -> retract)
        3. Increment current_box_index
        4. Trigger 'cycle_complete' if all boxes placed, else 'finished_placing' to pick next
        """
        try:
            # Check if stop has been requested
            if self._stop_requested.is_set():
                self._stop_requested.clear()
                print("Place operation cancelled due to stop request")
                return
            
            # Check if we have place positions available
            if not self.context.place_positions:
                self.fault("No place positions available - configure the palletizer first with /configure")
                return
            
            # Check if we have more boxes to place
            if self.context.current_box_index >= len(self.context.place_positions):
                self.fault("Box index out of range - all place positions have been used")
                return
            
            # Get the next place position (already in robot frame, in mm)
            place_position = self.context.place_positions[self.context.current_box_index]
            print(f"Placing box {self.context.current_box_index + 1}: position (robot frame) {place_position} mm")
            
            # Convert from mm to meters for motion controller
            place_position_m = np.array(place_position) / 1000.0
            
            # Execute place sequence using motion controller
            if not self.motion_controller.move_to_place(list(place_position_m)):
                self.fault("Place motion failed")
                return
            
            # Increment the box index after successful placement
            self.context.current_box_index += 1
            print(f"Successfully placed box {self.context.current_box_index}")
            
            # Check if we've placed all boxes
            if self.context.current_box_index >= self.context.total_boxes:
                print(f"All {self.context.total_boxes} boxes placed successfully!")
                self.trigger('cycle_complete')
            else:
                # More boxes to pick and place
                print(f"Ready to pick next box ({self.context.current_box_index + 1}/{self.context.total_boxes})")
                self.trigger('finished_placing')
            
        except Exception as e:
            self.fault(f"Place error: {str(e)}")
    
    @on_state_change
    def on_any_state_change(self, old_state: str, new_state: str, trigger: str):
        """Called on every state transition. Useful for logging."""
        print(f"State change: {old_state} -> {new_state} on trigger '{trigger}'")
        if self._stop_requested.is_set():
            print("Note: stop has been requested, current operation will be cancelled at the next check")
