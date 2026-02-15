import csv
import numpy as np
import matplotlib.pyplot as plt

csv_file = "motion_log.csv"

# Lists to store positions
cartesian_positions = []

with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        motion_type = row["Motion Type"]
        pos_str = row["Position/Joints"]
        pos = np.array([float(x) for x in pos_str.split(",")])

        # Only include moveL (Cartesian) positions
        if motion_type == "moveL":
            cartesian_positions.append(pos[:3])  # x, y, z

# Convert to NumPy array
cartesian_positions = np.array(cartesian_positions)

# Add robot base at origin
robot_base = np.array([[0, 0, 0]])
cartesian_positions_with_base = np.vstack([robot_base, cartesian_positions])

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(cartesian_positions_with_base[:, 0],
        cartesian_positions_with_base[:, 1],
        cartesian_positions_with_base[:, 2],
        marker='o', color='blue', label='Robot Path')

# Add labels for each point to show order (starting from 0 = robot base)
for i, (x, y, z) in enumerate(cartesian_positions_with_base):
    ax.text(x, y, z, str(i), color='red', fontsize=9)

# Highlight robot base with a different marker
ax.scatter(0, 0, 0, color='green', s=100, label='Robot Base')

ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_title('Robot Linear Path (moveL) with Robot Base')
ax.legend()
ax.grid(True)

plt.show()
