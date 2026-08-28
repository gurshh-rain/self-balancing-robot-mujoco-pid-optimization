import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from skopt import gp_minimize

# Absolute path resolution for model file
script_dir = Path(__file__).resolve().parent
xml_path = script_dir / "models" / "inverted_pendulum.xml"

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

left_motor_id = model.actuator("left_motor").id
right_motor_id = model.actuator("right_motor").id
chassis_id = model.body("chassis").id

dt = model.opt.timestep

def get_chassis_pitch(model, data):
    R = data.xmat[chassis_id].reshape(3, 3)
    return np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 1]**2))

# Global reference to viewer so the RL loop can render live steps
viewer_handle = None
trial_counter = 0

def evaluate_pid_visually(gains):
    global trial_counter
    trial_counter += 1
    
    Kp_pitch, Ki_pitch, Kd_pitch = gains
    print(f"\n--- [Iteration {trial_counter}] Testing Gains: Kp={Kp_pitch:.1f}, Ki={Ki_pitch:.2f}, Kd={Kd_pitch:.1f} ---")
    
    # Outer position loop gains
    Kp_pos = 0.12
    Kd_pos = 0.25

    mujoco.mj_resetData(model, data)
    
    # Reset robot state: Spawn at center with forward jolt
    data.qpos[0] = 0.0   # X position
    data.qpos[2] = 0.08  # Z height (Wheels on ground)
    data.qvel[0] = 1.2   # Initial forward jolt impulse (1.2 m/s)
    
    mujoco.mj_forward(model, data)

    integral_pitch_error = 0.0
    target_pitch_filtered = 0.0
    total_cost = 0.0
    
    episode_steps = 300  # ~1.5 seconds per iteration episode
    
    for step in range(episode_steps):
        step_start = time.time()

        current_x = data.qpos[0]
        current_vx = data.qvel[0]
        current_pitch = get_chassis_pitch(model, data)
        pitch_velocity = data.qvel[4]

        # Early Termination Penalty: Robot fell over completely (~45 degrees)
        if abs(current_pitch) > 0.8:
            total_cost += 5000.0 + (episode_steps - step) * 10.0
            print("❌ Tipped Over! High Penalty Assigned.")
            break

        # Outer Loop: Position to Target Pitch
        pos_error = 0.0 - current_x
        raw_target_pitch = (Kp_pos * pos_error) - (Kd_pos * current_vx)
        raw_target_pitch = np.clip(raw_target_pitch, -0.14, 0.14)
        target_pitch_filtered = (0.95 * target_pitch_filtered) + (0.05 * raw_target_pitch)

        # Inner Loop: Pitch PID Controller
        pitch_error = target_pitch_filtered - current_pitch
        integral_pitch_error += pitch_error * dt
        integral_pitch_error = np.clip(integral_pitch_error, -1.0, 1.0)

        torque = -1.0 * (
            (Kp_pitch * pitch_error) + 
            (Ki_pitch * integral_pitch_error) - 
            (Kd_pitch * pitch_velocity)
        )
        torque = np.clip(torque, -8.0, 8.0)

        data.ctrl[left_motor_id] = torque
        data.ctrl[right_motor_id] = torque

        mujoco.mj_step(model, data)

        # Sync visual frame to interactive window
        if viewer_handle is not None and viewer_handle.is_running():
            viewer_handle.sync()

        # Step-wise Cost: Penalizes position drift from 0, tilt angle, and motor effort
        step_cost = (current_x ** 2) * 15.0 + (current_pitch ** 2) * 60.0 + (torque ** 2) * 0.01
        total_cost += step_cost

        # Pacing to keep physics at real-time speed for human viewing
        time_to_sleep = dt - (time.time() - step_start)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    print(f"Episode Finish. Cost: {total_cost:.1f}")
    return total_cost

# -------------------------------------------------------------
# MAIN TRAINER LOOP WITH LIVE WINDOW
# -------------------------------------------------------------
print("Launching Live Visual PID RL Optimization...")

# Open native viewer window
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer_handle = viewer
    
    search_space = [
        (10.0, 90.0),  # Kp bounds
        (0.0, 3.0),    # Ki bounds
        (0.5, 10.0)    # Kd bounds
    ]

    # Run Bayesian Optimization over 12 iterations
    result = gp_minimize(
        func=evaluate_pid_visually,
        dimensions=search_space,
        n_calls=50,
        random_state=42
    )

    print("\n==========================================")
    print("      RL OPTIMIZATION COMPLETE!           ")
    print("==========================================")
    print(f"Best Kp: {result.x[0]:.2f}")
    print(f"Best Ki: {result.x[1]:.2f}")
    print(f"Best Kd: {result.x[2]:.2f}")
    print("==========================================")
    
    # Run a final demonstration run using the winning gains
    print("\nRunning final victory trial using best learned gains...")
    evaluate_pid_visually(result.x)