I make some assumption while programming this paletizer. 

First the camera mounting posistion is at 800mm in the Z axis and the mocked detections are at 0mm in Z. That makes the box to grab at a height of 800mm. It does not make sens since the height reach of the arm is 850 mm. So, i assume that the 0mm in Z from mocked detection is interpreted in teh arm reference frame.

I am also asuming that the detection orientation are in the robot frame because to fully convert from camera to robot frame i would need a roll and pitch angle too. But it makes sens to have a common yaw axis because the box should be flat on the conveyor.