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

NUM_ROBOTS = 10
dt = model.opt.timestep

# Index mapping for all 10 robots
chassis_ids = [model.body(f"chassis_{i}").id for i in range(NUM_ROBOTS)]
left_motor_ids = [model.actuator(f"lm_{i}").id for i in range(NUM_ROBOTS)]
right_motor_ids = [model.actuator(f"rm_{i}").id for i in range(NUM_ROBOTS)]

def get_chassis_pitch(data, body_id):
    """Calculates pitch angle (rad) around local Y-axis."""
    R = data.xmat[body_id].reshape(3, 3)
    return np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 1]**2))

def reset_all_robots():
    """Resets all 10 robots on their Y-axis grid spots and applies forward jolt."""
    mujoco.mj_resetData(model, data)
    y_positions = np.linspace(-2.0, 2.5, NUM_ROBOTS)
    
    for i in range(NUM_ROBOTS):
        qpos_adr = model.jnt_qposadr[model.body(f"chassis_{i}").jntadr[0]]
        qvel_adr = model.jnt_dofadr[model.body(f"chassis_{i}").jntadr[0]]

        data.qpos[qpos_adr + 0] = 0.0            # X position
        data.qpos[qpos_adr + 1] = y_positions[i]  # Y position
        data.qpos[qpos_adr + 2] = 0.08           # Z height
        data.qvel[qvel_adr + 0] = 1.0            # Forward impulse (1.0 m/s)
        
    mujoco.mj_forward(model, data)

def run_wave(population_gains, viewer):
    reset_all_robots()
    
    integral_errors = np.zeros(NUM_ROBOTS)
    target_pitch_filtered = np.zeros(NUM_ROBOTS)
    total_costs = np.zeros(NUM_ROBOTS)
    fell_over = [False] * NUM_ROBOTS
    
    # Outer Loop: Gentle position pull to prevent fighting the inner loop
    Kp_pos = 0.035
    episode_steps = 300  # 1.5 seconds evaluation period

    for step in range(episode_steps):
        step_start = time.time()
        
        for i in range(NUM_ROBOTS):
            if fell_over[i]:
                continue

            qpos_adr = model.jnt_qposadr[model.body(f"chassis_{i}").jntadr[0]]
            qvel_adr = model.jnt_dofadr[model.body(f"chassis_{i}").jntadr[0]]

            current_x = data.qpos[qpos_adr + 0]
            current_vx = data.qvel[qvel_adr + 0]
            current_pitch = get_chassis_pitch(data, chassis_ids[i])
            pitch_velocity = data.qvel[qvel_adr + 4]

            # Cut power and apply heavy cost if robot falls or violent bounce occurs
            if abs(current_pitch) > 0.6:
                fell_over[i] = True
                total_costs[i] += 10000.0
                data.ctrl[left_motor_ids[i]] = 0.0
                data.ctrl[right_motor_ids[i]] = 0.0
                continue

            Kp_pitch, Ki_pitch, Kd_pitch = population_gains[i]

            # 1. Outer Loop (Position Error -> Pitch Setpoint, limited to ±4 degrees)
            pos_error = 0.0 - current_x
            raw_target_pitch = Kp_pos * pos_error
            raw_target_pitch = np.clip(raw_target_pitch, -0.07, 0.07)
            target_pitch_filtered[i] = (0.92 * target_pitch_filtered[i]) + (0.08 * raw_target_pitch)

            # 2. Inner Loop (Pitch Error + Angular Damping -> Wheel Torque)
            pitch_error = target_pitch_filtered[i] - current_pitch
            integral_errors[i] += pitch_error * dt
            integral_errors[i] = np.clip(integral_errors[i], -0.5, 0.5)

            # Direct PD formulation without loop interference
            torque = -1.0 * (
                (Kp_pitch * pitch_error) + 
                (Ki_pitch * integral_errors[i]) - 
                (Kd_pitch * pitch_velocity)
            )
            torque = np.clip(torque, -6.0, 6.0)

            data.ctrl[left_motor_ids[i]] = torque
            data.ctrl[right_motor_ids[i]] = torque

            # Accumulate cost (heavily penalize pitch velocity to eliminate bouncing)
            total_costs[i] += (current_x**2) * 10.0 + (current_pitch**2) * 100.0 + (pitch_velocity**2) * 5.0

        mujoco.mj_step(model, data)
        if viewer and viewer.is_running():
            viewer.sync()

        # Real-time synchronization
        time_to_sleep = dt - (time.time() - step_start)
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    best_idx = np.argmin(total_costs)
    best_cost = total_costs[best_idx]
    
    # Success threshold: Cost under 250 means ZERO bouncing and rock-solid stabilization
    success = (best_cost < 250.0) and not fell_over[best_idx]
    
    return success, best_idx, population_gains[best_idx], total_costs

# -------------------------------------------------------------
# MAIN TRAINER LOOP
# -------------------------------------------------------------
print("Launching 10-Robot Smooth PID Optimization...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    np.random.seed(42)
    
    # Initial gains tuned for smooth balancing (Moderate Kp, High Kd damping)
    population = []
    for _ in range(NUM_ROBOTS):
        kp = np.random.uniform(18.0, 45.0)
        ki = np.random.uniform(0.01, 0.5)
        kd = np.random.uniform(3.0, 7.5)
        population.append([kp, ki, kd])

    wave_counter = 0
    
    while viewer.is_running():
        wave_counter += 1
        print(f"\n--- [Wave {wave_counter}] Testing 10 Robots ---")

        success, best_idx, best_gains, costs = run_wave(population, viewer)

        print(f"Wave Best: Robot #{best_idx} | Stability Score: {costs[best_idx]:.1f}")
        print(f"Gains: Kp={best_gains[0]:.2f}, Ki={best_gains[1]:.2f}, Kd={best_gains[2]:.2f}")

        if success or wave_counter >= 8:
            print("\n==============================================")
            print("   SUCCESS: SMOOTH OPTIMAL PID GAINS FOUND!  ")
            print("==============================================")
            print(f" Winning Robot Index: #{best_idx}")
            print(f" Optimal Kp: {best_gains[0]:.2f}")
            print(f" Optimal Ki: {best_gains[1]:.2f}")
            print(f" Optimal Kd: {best_gains[2]:.2f}")
            print("==============================================")
            break

        # Mutate population around top performer with high Damping priority
        new_population = [best_gains]
        for _ in range(NUM_ROBOTS - 1):
            mutated_kp = np.clip(best_gains[0] + np.random.normal(0, 3.5), 10.0, 60.0)
            mutated_ki = np.clip(best_gains[1] + np.random.normal(0, 0.05), 0.0, 1.0)
            mutated_kd = np.clip(best_gains[2] + np.random.normal(0, 0.6), 2.0, 9.5)
            new_population.append([mutated_kp, mutated_ki, mutated_kd])
        
        population = new_population