import pygame 
import math
import numpy as np
import sounddevice as sd
import threading 
import sys

# --- SIMULATION PARAMETERS (CONSTANTS) ---
g = 9.81  # Gravity (m/s^2)
m1, m2, m3 = 1.0, 1.0, 1.0  # Masses (kg)
dt = 0.005 # Time step (seconds)

# PENDULUM A (REFERENCE) - All rod lengths set to 1.0 m (Blue)
L_A = np.array([1.0, 1.0, 1.0]) 
M_A = np.array([m1, m2, m3])

# PENDULUM B (CHAOTIC TWIN - All rods set to 1.0 m) (Red)
# Note: L_B[2] is now 2*m3 to make this system dynamically different from A.
L_B = np.array([1.0, 1.0, 1.0]) 
M_B = np.array([m1, m2, m3])

# --- DISPLAY PARAMETERS (ADJUSTED FOR LARGER SCREEN AND MORE SPACE) ---
SCALE = 100.0  # Scale factor for display
WIDTH, HEIGHT = 1400, 600 # Increased window size
ORIGIN_Y = HEIGHT // 5

# Origin for Pendulum A (Left - spaced further out)
ORIGIN_A = (WIDTH // 5, ORIGIN_Y)
# Origin for Pendulum B (Right - spaced further out)
ORIGIN_B = (4 * WIDTH // 5, ORIGIN_Y)

# --- ENERGY CALCULATION ---
def calculate_energy(Y_in, L_in, M_in):
    """Calculates the total energy for a single triple pendulum system."""
    th = Y_in[:3]
    om = Y_in[3:]

    V = 0.0 # Potential Energy
    K = 0.0 # Kinetic Energy
    
    for i in range(3):
        # Position is relative to origin
        x_i = sum(L_in[k] * math.sin(th[k]) for k in range(i + 1))
        y_i = sum(L_in[k] * math.cos(th[k]) for k in range(i + 1))
        
        # Velocity components derived from angular velocity
        vx_i = sum(L_in[k] * om[k] * math.cos(th[k]) for k in range(i + 1))
        vy_i = sum(L_in[k] * om[k] * -math.sin(th[k]) for k in range(i + 1))
        
        # Update Potential Energy (V = -mgy)
        V += -M_in[i] * g * y_i
        
        # Update Kinetic Energy (K = 0.5 * m * (vx^2 + vy^2))
        K += 0.5 * M_in[i] * (vx_i**2 + vy_i**2)

    return K + V

# --- STATE VECTOR Y (initial state) ---
# Y = [theta1, theta2, theta3, omega1, omega2, omega3]
# Initial state set to all zeros for easy adjustment upon pause

#Y_initial_A = np.array([0, 0, 0, 0, 0, 0], dtype=float)    
#Y_initial_B = np.array([0, 0, 0, 0, 0, 0], dtype=float)

Y_initial_A = np.array([0.7854, 0.7854, 0.7854 , 0.0, 0.0, 0.0], dtype=float)    
Y_initial_B = np.array([0, -1.5708, 0.0, 0.0, 0.0, 0.0], dtype=float)

# Global State now holds two pendulums: Y[0] is A, Y[1] is B
Y = [Y_initial_A, Y_initial_B]

paused = True
initial_energy = calculate_energy(Y[0], L_A, M_A) 

# Threading Lock for safe state access
state_lock = threading.Lock()

# --- AUDIO & EDITING CONTROL VARIABLES ---
# 0: Pendulum A, 1: Pendulum B, 2: BOTH (New state for 'C' key)
audio_source_index = 0 
editing_pendulum_index = 0 # 0 for A (Blue), 1 for B (Red)

# --- AUDIO GENERATOR ---

sample_rate = 44100
blocksize = 4096
max_amp = 0.15 
channels = 2
phase_A = 0.0 # Separate phase for A
phase_B = 0.0 # Separate phase for B

# Audio Mapping Constants
FREQ_BASELINE = 200.0  
FREQ_MAX_RANGE = 1800.0
BASE_AMP = 0.05        
MOD_FACTOR = 0.8        
DUAL_VOL_SCALE = 0.8 # Scale volume down when both are active

def wrap(angle):
    """Wraps any angle in radians to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Function to map pendulum state to sound (returns [frequency, amp_L, amp_R])
def map_state_to_sound(Y_snapshot):
    th0, th1, th2_raw = Y_snapshot[0], Y_snapshot[1], Y_snapshot[2]

    # --- FIX: Wrap th2 for frequency calculation ---
    # The raw th2 angle (th2_raw) can grow infinitely large.
    # We must wrap it to the [-pi, pi] range to get the
    # actual displacement from the vertical for sonification.
    th2 = wrap(th2_raw)
    # --- END FIX ---
    
    # 1. Frequency (linked to the angular position of the bottom bob, theta3)
    # Now, abs(th2) is correctly in the [0, pi] range,
    # so dividing by math.pi gives the desired [0, 1] factor.
    displacement_factor = np.clip(abs(th2) / math.pi, 0.02, 1.0) 
    freq = displacement_factor * FREQ_MAX_RANGE 
    
    # 2. Amplitudes (linked to the angular position of bobs 1 and 2)
    # These lines were already working correctly because math.sin() is
    # periodic, so math.sin(th0) is identical to math.sin(wrap(th0)).
    amp_left = BASE_AMP + MOD_FACTOR * abs(math.sin(th0))
    amp_right = BASE_AMP + MOD_FACTOR * abs(math.sin(th1))
    
    return freq, amp_left, amp_right

def audio_callback(outdata, frames, time_info, status):
    global Y, phase_A, phase_B, state_lock, audio_source_index
    
    tbuf = np.zeros((frames, channels), dtype=np.float32)

    with state_lock:
        Y_snapshot_A = Y[0].copy()
        Y_snapshot_B = Y[1].copy()
        current_audio_source = audio_source_index
            
    audio_scale = max_amp
    is_dual_source = (current_audio_source == 2)
    
    if is_dual_source:
        audio_scale *= DUAL_VOL_SCALE

    if current_audio_source == 0 or is_dual_source:
        # Map Pendulum A state to sound parameters
        freq_A, amp_L_A, amp_R_A = map_state_to_sound(Y_snapshot_A)

    if current_audio_source == 1 or is_dual_source:
        # Map Pendulum B state to sound parameters
        freq_B, amp_L_B, amp_R_B = map_state_to_sound(Y_snapshot_B)
    
    # Generate the sine wave buffer
    for i in range(frames):
        sample_L = 0.0
        sample_R = 0.0
        
        if current_audio_source == 0:
            # Single A Source
            phase_A += (2.0 * math.pi * freq_A) / sample_rate
            if phase_A > 2.0 * math.pi: phase_A -= 2.0 * math.pi
            
            sample = math.sin(phase_A)
            sample_L = sample * amp_L_A
            sample_R = sample * amp_R_A
            
        elif current_audio_source == 1:
            # Single B Source
            phase_B += (2.0 * math.pi * freq_B) / sample_rate
            if phase_B > 2.0 * math.pi: phase_B -= 2.0 * math.pi
            
            sample = math.sin(phase_B)
            sample_L = sample * amp_L_B
            sample_R = sample * amp_R_B
            
        elif current_audio_source == 2:
            # Dual Source: Sum of two separate sine waves
            phase_A += (2.0 * math.pi * freq_A) / sample_rate
            if phase_A > 2.0 * math.pi: phase_A -= 2.0 * math.pi
            
            phase_B += (2.0 * math.pi * freq_B) / sample_rate
            if phase_B > 2.0 * math.pi: phase_B -= 2.0 * math.pi
            
            
            sample_L = math.sin(phase_A) * amp_L_A  + math.sin(phase_B) * amp_L_B 
            sample_R = math.sin(phase_A) * amp_R_A + math.sin(phase_B) * amp_R_B 
            
        tbuf[i, 0] = sample_L
        tbuf[i, 1] = sample_R
        
    tbuf *= audio_scale
    outdata[:] = tbuf
    

# --- CORE PHYSICS: FULL COUPLED MATRIX SOLUTION ---
def get_derivatives(Y_in, L_in, M_in):
    """Calculates the derivatives for a single triple pendulum system."""
    th = Y_in[:3]
    om = Y_in[3:]
    
    # 1. Mass Matrix (M_matrix * a = F_vector)
    M_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            sum_m = np.sum(M_in[max(i, j):])
            M_matrix[i, j] = sum_m * L_in[i] * L_in[j] * math.cos(th[i] - th[j])

    # 2. Force Vector (F_vector)
    F_vector = np.zeros(3)
    for i in range(3):
        # Term 1: Centripetal/Coriolis forces
        term1 = 0.0
        for j in range(3):
            sum_m = np.sum(M_in[max(i, j):])
            term1 += sum_m * L_in[i] * L_in[j] * math.sin(th[i] - th[j]) * om[j]**2
        
        # Term 2: Gravity 
        sum_m_gravity = np.sum(M_in[i:])
        term2 = g * math.sin(th[i]) * sum_m_gravity * L_in[i]
        
        F_vector[i] = -term1 - term2

    # 3. Solve for Accelerations (a = M_matrix^-1 * F_vector)
    try:
        a = np.linalg.solve(M_matrix, F_vector)
    except np.linalg.LinAlgError:
        # Handle singular matrix (e.g., if all angles are near 0 or pi)
        a = np.zeros(3) 

    # 4. Construct Derivative Vector Y_dot
    Y_dot = np.zeros(6, dtype=float)
    Y_dot[:3] = om  # d(th)/dt = omegas
    Y_dot[3:] = a   # d(om)/dt = accelerations
    
    return Y_dot

# --- RK4 INTEGRATOR ---
def rk4_step(Y_current, L_current, M_current, h):
    """Performs one step of Runge-Kutta 4 integration for a single pendulum."""
    k1 = h * get_derivatives(Y_current, L_current, M_current)
    k2 = h * get_derivatives(Y_current + 0.5 * k1, L_current, M_current)
    k3 = h * get_derivatives(Y_current + 0.5 * k2, L_current, M_current)
    k4 = h * get_derivatives(Y_current + k3, L_current, M_current)
    
    Y_next = Y_current + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return Y_next
    
# --- PYGAME SETUP ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dual Triple Pendulum Chaos Simulator (A vs B)")
clock = pygame.time.Clock()
FPS = 144

font = pygame.font.Font(None, 24)

# Load the custom speaker image
# IMPORTANT: Make sure 'speaker_icon.png' is in the same directory as this script.
try:
    original_speaker_image = pygame.image.load('speaker_icon.png').convert_alpha()
    # Adjust scaling as needed, a smaller scale here for a smaller icon
    SPEAKER_IMAGE_SCALE_FACTOR = 0.15 # Adjust this value to make the image smaller or larger
    
    # Calculate new dimensions for the scaled image
    img_width, img_height = original_speaker_image.get_size()
    scaled_width = int(img_width * SPEAKER_IMAGE_SCALE_FACTOR)
    scaled_height = int(img_height * SPEAKER_IMAGE_SCALE_FACTOR)
    
    # Scale the image
    scaled_speaker_image = pygame.transform.scale(original_speaker_image, (scaled_width, scaled_height))
    
    # Adjust offset to position it significantly below the pivot
    # ORIGIN_Y is where the pivot of the pendulum is.
    # The image will be centered horizontally below this pivot.
    SPEAKER_OFFSET_Y = 400 # Increased offset to move it further down
    
except pygame.error as e:
    print(f"Error loading speaker_icon.png: {e}. Falling back to default polygon icon.", file=sys.stderr)
    scaled_speaker_image = None # Indicate that image loading failed
    
    # Fallback to the original polygon if image loading fails
    # Simple Speaker Icon (List of points for a polygon)
    # Centered at (0, 0), scaled later
    SPEAKER_ICON_POINTS = [
        (0, 0), (-10, 5), (-10, 15), (0, 20), 
        (0, 15), (5, 12), (5, 8), (0, 5)
    ]
    SPEAKER_ICON_SCALE = 1.2
    SPEAKER_ICON_COLOR = (255, 255, 0)
    SPEAKER_OFFSET_Y = 250 # Still apply the larger offset

# Trails track the final bob of both Pendulums
trail_A = [] # Blue Trail (Reference)
trail_B = [] # Red Trail (Chaotic Twin)
MAX_TRAIL_LENGTH = 20000

# --- Start the audio stream ONCE ---
try:
    audio_stream = sd.OutputStream(channels=channels, samplerate=sample_rate, 
                                   blocksize=blocksize, callback=audio_callback)
    audio_stream.start()
except Exception as e:
    print(f"Warning: Error starting audio stream ({e}). Running simulation silently.", file=sys.stderr)
    audio_stream = None

# --- MAIN LOOP ---
running = True

while running:
    # 1. EVENT HANDLING (Handles angle changes while paused)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            
            # Toggle pause on spacebar
            elif e.key == pygame.K_SPACE:
                paused = not paused
                if not paused:
                    # When starting, reset the initial energy baseline
                    initial_energy = calculate_energy(Y[0], L_A, M_A)
                    trail_A = [] # Clear trail A on start/unpause
                    trail_B = [] # Clear trail B on start/unpause
                    print(f"Simulation Started. New Reference Energy: {initial_energy:.8f} J")
                else:
                    print(f"Simulation Paused. Editing is set to Pendulum {'A' if editing_pendulum_index == 0 else 'B'}")

            # --- SELECT PENDULUM FOR EDITING AND AUDIO ---
            elif e.key == pygame.K_a:
                with state_lock:
                    audio_source_index = 0 
                editing_pendulum_index = 0
                print("Selected Pendulum A (Blue) for Audio and Angle Editing.")
            elif e.key == pygame.K_b:
                with state_lock:
                    audio_source_index = 1 
                editing_pendulum_index = 1
                print("Selected Pendulum B (Red) for Audio and Angle Editing.")
            elif e.key == pygame.K_c:
                with state_lock:
                    audio_source_index = 2 # Both
                # Keep editing on the current index
                print(f"Selected BOTH Pendulums for Audio. Editing remains on Pendulum {'A' if editing_pendulum_index == 0 else 'B'}")
            # -----------------------------------------------

            # Adjust angles (Applies ONLY to the selected pendulum)
            if paused:
                idx = editing_pendulum_index
                with state_lock:
                    angle_change = 0
                    if e.key == pygame.K_KP9: angle_change = math.pi / 60
                    elif e.key == pygame.K_KP7: angle_change = -math.pi / 60
                    
                    if angle_change != 0:
                        Y[idx][0] += angle_change # Change T1
                    
                    angle_change = 0
                    if e.key == pygame.K_KP6: angle_change = math.pi / 60
                    elif e.key == pygame.K_KP4: angle_change = -math.pi / 60
                    
                    if angle_change != 0:
                        Y[idx][1] += angle_change # Change T2

                    angle_change = 0
                    if e.key == pygame.K_KP3: angle_change = math.pi / 60
                    elif e.key == pygame.K_KP1: angle_change = -math.pi / 60
                    
                    if angle_change != 0:
                        Y[idx][2] += angle_change # Change T3
                    
                    # Ensure angles are wrapped
                    Y[idx][:3] = np.array([wrap(angle) for angle in Y[idx][:3]])

    
    # 2. PHYSICS UPDATE
    if not paused: 
        # --- Runge-Kutta 4 Integration for BOTH Pendulums ---
        with state_lock:
            # Pendulum A
            Y[0] = rk4_step(Y[0], L_A, M_A, dt)
            # Pendulum B
            Y[1] = rk4_step(Y[1], L_B, M_B, dt)
        
        # Calculate current energy and drift for Pendulum A (Reference)
        current_energy = calculate_energy(Y[0], L_A, M_A)
        energy_drift_abs = abs(current_energy - initial_energy)
        energy_drift_percent = (energy_drift_abs / abs(initial_energy) * 100) if abs(initial_energy) > 1e-9 else 0.0

    else:
        # Recalculate energy while paused to show accurate drift when unpausing
        current_energy = calculate_energy(Y[0], L_A, M_A)
        energy_drift_percent = 0.0
    
    # 3. CONSOLE PRINTING LOGIC (RUNS EVERY FRAME)
    th_A = Y[0][:3]
    th_B = Y[1][:3]
    
    # Format the output clearly: Angles are in Radians (theta1, theta2, theta3)
    # Using the carriage return '\r' to overwrite the previous line in a VS Code terminal
    output = (
        f"\rA (1.0m, m1,m2,m3) Angles (rad): T1={th_A[0]:.4f}, T2={th_A[1]:.4f}, T3={th_A[2]:.4f} | "
        f"B (1.0m, m1,m2,m3) Angles (rad): T1={th_B[0]:.4f}, T2={th_B[1]:.4f}, T3={th_B[2]:.4f}   "
    )
    sys.stdout.write(output)
    sys.stdout.flush()
    # --- END CONSOLE PRINTING LOGIC ---

    # 4. COMPUTE DISPLAY POSITIONS & DRAWING SETUP
    
    screen.fill((15, 15, 25))
    
    # --- Color Definitions ---
    BLUE_ROD = (100, 100, 200)
    BLUE_BOB = (120, 120, 255)
    BLUE_TRAIL = (80, 100, 255)

    RED_ROD = (200, 100, 100)
    RED_BOB = (255, 120, 120)
    RED_TRAIL = (255, 100, 80)
    
    # Pendulum Definitions for Drawing: (State_Vector, Lengths, Rod_Color, Bob_Color, Origin, Trail_List, Trail_Color, Index)
    pendulums_data = [
        # Pendulum A (Reference) is BLUE
        (Y[0], L_A, BLUE_ROD, BLUE_BOB, ORIGIN_A, trail_A, BLUE_TRAIL, 0), 
        # Pendulum B (Chaotic Twin) is RED
        (Y[1], L_B, RED_ROD, RED_BOB, ORIGIN_B, trail_B, RED_TRAIL, 1) 
    ]
    
    # Draw pivots
    pygame.draw.circle(screen, (200, 200, 200), ORIGIN_A, 6)
    pygame.draw.circle(screen, (200, 200, 200), ORIGIN_B, 6)

    
    # DRAW PENDULUMS and UPDATE TRAILS
    for Y_current, L_current, rod_color, bob_color, origin, current_trail, trail_color, p_index in pendulums_data:
        th1, th2, th3 = Y_current[0], Y_current[1], Y_current[2]
        L1_c, L2_c, L3_c = L_current
        
        # Calculate rod positions relative to the current origin
        x1_rel = L1_c * math.sin(th1)
        y1_rel = L1_c * math.cos(th1)
        x2_rel = L2_c * math.sin(th2)
        y2_rel = L2_c * math.cos(th2)
        x3_rel = L3_c * math.sin(th3)
        y3_rel = L3_c * math.cos(th3)

        # Absolute positions (scaled and offset by origin)
        p1 = (origin[0] + x1_rel * SCALE, origin[1] + y1_rel * SCALE)
        p2 = (p1[0] + x2_rel * SCALE, p1[1] + y2_rel * SCALE)
        p3 = (p2[0] + x3_rel * SCALE, p2[1] + y3_rel * SCALE) # Final bob position

        # --- Update Trail ---
        if not paused:
            current_trail.append(p3)
        
        if len(current_trail) > MAX_TRAIL_LENGTH: 
            current_trail.pop(0)

        # --- Draw Trail ---
        if len(current_trail) > 2:
            pygame.draw.lines(screen, trail_color, False, current_trail, 1)

        # --- Draw Pendulum Rods and Bobs ---
        
        # rods
        pygame.draw.line(screen, rod_color, origin, p1, 2)
        pygame.draw.line(screen, rod_color, p1, p2, 2)
        pygame.draw.line(screen, rod_color, p2, p3, 2)

        # bobs
        pygame.draw.circle(screen, bob_color, (int(p1[0]), int(p1[1])), 8)
        pygame.draw.circle(screen, bob_color, (int(p2[0]), int(p2[1])), 8)
        pygame.draw.circle(screen, bob_color, (int(p3[0]), int(p3[1])), 8)
        
        # --- Draw Speaker Icon if this is the active audio source ---
        if audio_source_index == p_index or audio_source_index == 2:
            if scaled_speaker_image:
                # Calculate icon position (below the main pivot)
                # Centered horizontally, at ORIGIN_Y + SPEAKER_OFFSET_Y
                icon_x = origin[0] - scaled_speaker_image.get_width() // 2
                icon_y = ORIGIN_Y + SPEAKER_OFFSET_Y
                icon_pos = (icon_x, icon_y)
                
                screen.blit(scaled_speaker_image, icon_pos)
            else:
                # Fallback to drawing the polygon if image failed to load
                icon_pos = (origin[0], ORIGIN_Y + SPEAKER_OFFSET_Y)
                transformed_points = []
                for px, py in SPEAKER_ICON_POINTS:
                    transformed_points.append((
                        icon_pos[0] + px * SPEAKER_ICON_SCALE, 
                        icon_pos[1] + py * SPEAKER_ICON_SCALE
                    ))
                pygame.draw.polygon(screen, SPEAKER_ICON_COLOR, transformed_points)
    
    
    # 5. Status/Instructions text
    energy_text = f"E DRIFT (A): {energy_drift_percent:.6f}%"
    
    audio_map = {0: "A (Blue)", 1: "B (Red)", 2: "BOTH (A & B)"}
    audio_src_text = audio_map.get(audio_source_index, "OFF")
    
    editing_src_text = "A (Blue)" if editing_pendulum_index == 0 else "B (Red)"

    if paused:
        status_text = (f"PAUSED: EDITING {editing_src_text} (A/B to Switch) | "
                       f"NUMPAD to Adjust Angles | SPACE to Start")
    else:
        status_text = (f"RUNNING | AUDIO: {audio_src_text} (Press A/B/C to Switch) | "
                       f"{energy_text}")

    text_surface = font.render(status_text, True, (200, 200, 200))
    screen.blit(text_surface, (10, 10))
    
    # Add Angle Adjustment Instructions
    edit_instr_text = "T1: 7/9 | T2: 4/6 | T3: 1/3 (Numpad)"
    edit_instr_surface = font.render(edit_instr_text, True, (150, 150, 150))
    screen.blit(edit_instr_surface, (10, 35))

    # 6. FLIP AND TICK
    pygame.display.flip()
    clock.tick(FPS)

# --- Cleanup ---
if audio_stream is not None:
    audio_stream.stop()
    audio_stream.close()

pygame.quit()
