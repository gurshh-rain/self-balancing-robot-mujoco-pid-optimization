import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

# Absolute path resolution for model file
script_dir = Path(__file__).resolve().parent
xml_path = script_dir / "models" / "inverted_pendulum.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# Actuator & Joint IDs
left_motor_id = model.actuator("left_motor").id
right_motor_id = model.actuator("right_motor").id
chassis_id = model.body("chassis").id

# --- CASCADED CONTROL GAINS ---

# Outer Loop: Position Control (Maps X error -> Target Pitch)
Kp_pos = 0.12        # Reduced so it doesn't over-react
Kd_pos = 0.25        # Velocity damping on position

# Inner Loop: Pitch Control (Maps Pitch error -> Wheel Torque)
Kp_pitch = 0      # Smooth Proportional gain
Ki_pitch = 0       # Integral gain for steady-state offset
Kd_pitch = 0       # Damping gain acting directly on angular velocity

integral_pitch_error = 0.0
target_pitch_filtered = 0.0
dt = model.opt.timestep

# Utility function to convert rotation matrix to Pitch (Y-axis rotation)
def get_chassis_pitch(data, body_id):
    R = data.xmat[body_id].reshape(3, 3)
    return np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 1]**2))

# --- RESET SIMULATION STATE ---
mujoco.mj_resetData(model, data)

# Spawn robot at X=0, sitting on ground
data.qpos[0] = 0.0   # X position
data.qpos[1] = 0.0   # Y position
data.qpos[2] = 0.08  # Z height (Wheels touching ground)

# JOLT FEATURE: Apply smooth initial linear impulse (Vx = 1.0 m/s)
data.qvel[0] = 1.0   

mujoco.mj_forward(model, data)

# Launch viewer using mjpython
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # -------------------------------------------------------------
        # 1. READ RAW SENSOR STATES (No Numerical Differentiation Needed)
        # -------------------------------------------------------------
        current_x = data.qpos[0]                         # Robot X position
        current_vx = data.qvel[0]                        # Forward velocity (m/s)
        current_pitch = get_chassis_pitch(data, chassis_id) # Pitch angle (rad)
        
        # Freejoint rotational velocity index 4 corresponds to Y-axis pitch velocity (rad/s)
        pitch_velocity = data.qvel[4]                    

        # -------------------------------------------------------------
        # 2. OUTER LOOP (Position -> Smooth Target Pitch)
        # -------------------------------------------------------------
        pos_error = 0.0 - current_x
        raw_target_pitch = (Kp_pos * pos_error) - (Kd_pos * current_vx)
        
        # Clamp maximum allowable tilt target (±8 degrees = ±0.14 rad)
        raw_target_pitch = np.clip(raw_target_pitch, -0.14, 0.14)

        # Low-pass filter target pitch to eliminate jitter/sharp inputs (Alpha = 0.05)
        target_pitch_filtered = (0.95 * target_pitch_filtered) + (0.05 * raw_target_pitch)

        # -------------------------------------------------------------
        # 3. INNER LOOP (Pitch -> Motor Torque)
        # -------------------------------------------------------------
        pitch_error = target_pitch_filtered - current_pitch
        
        # Accumulate integral with tight anti-windup clamping
        integral_pitch_error += pitch_error * dt
        integral_pitch_error = np.clip(integral_pitch_error, -1.0, 1.0)

        # PID Output: Derivative term uses clean physical pitch_velocity directly
        torque = -1.0 * (
            (Kp_pitch * pitch_error) + 
            (Ki_pitch * integral_pitch_error) - 
            (Kd_pitch * pitch_velocity)
        )

        # Smooth torque clamping
        torque = np.clip(torque, -8.0, 8.0)

        # -------------------------------------------------------------
        # 4. STEP PHYSICS
        # -------------------------------------------------------------
        data.ctrl[left_motor_id] = torque
        data.ctrl[right_motor_id] = torque

        mujoco.mj_step(model, data)
        viewer.sync()

        # Maintain real-time simulation pacing
        time_to_sleep = dt - (time.time() - step_start)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)