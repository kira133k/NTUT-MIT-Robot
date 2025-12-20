import numpy as np
import mujoco
import mujoco.viewer
import time
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.linalg
from pynput import keyboard

"""

<<------------- Sensor ------------->>
sensor_index: 0 , name: L_thigh_pos , dim: 1
sensor_index: 1 , name: L_calf_pos , dim: 1
sensor_index: 2 , name: L_wheel_pos , dim: 1
sensor_index: 3 , name: R_thigh_pos , dim: 1
sensor_index: 4 , name: R_calf_pos , dim: 1
sensor_index: 5 , name: R_wheel_pos , dim: 1
sensor_index: 6 , name: L_thigh_vel , dim: 1
sensor_index: 7 , name: L_calf_vel , dim: 1
sensor_index: 8 , name: L_wheel_vel , dim: 1
sensor_index: 9 , name: R_thigh_vel , dim: 1
sensor_index: 10 , name: R_calf_vel , dim: 1
sensor_index: 11 , name: R_wheel_vel , dim: 1
sensor_index: 12 , name: L_thigh_torque , dim: 1
sensor_index: 13 , name: L_calf_torque , dim: 1
sensor_index: 14 , name: L_wheel_torque , dim: 1
sensor_index: 15 , name: R_thigh_torque , dim: 1
sensor_index: 16 , name: R_calf_torque , dim: 1
sensor_index: 17 , name: R_wheel_torque , dim: 1
sensor_index: 18 , name: imu_quat , dim: 4
sensor_index: 22 , name: imu_gyro , dim: 3
sensor_index: 25 , name: imu_acc , dim: 3
sensor_index: 28 , name: frame_pos , dim: 3
sensor_index: 31 , name: frame_lin_vel , dim: 3
sensor_index: 34 , name: frame_ang_vel , dim: 3

"""

# ==========================================
# 1. Model Parameters
# ==========================================

# Robotic Parameters
M = 6.442               # Total mass(From pinocchio) [kg]
m = 0.28                # Single wheel mass [kg]
d = 0.3291              # Wheel track [m]
r = 0.07                # Wheel radius [m]
g = 9.81                # Gravity [m/s^2]

def LQR(current_l,pos=None):

    # Moment of inertia
    I_wheel = m * (r**2) / 2        # Wheel inertia around center
    J_p = M * (current_l**2) / 3            # Body inertia around CG
    J_delta = M * (d**2) / 12       # Body inertia due to wheel track

    # ==========================================
    # 2. LQR Controller
    # ==========================================

    term1 = (J_p + M * (current_l**2))
    term2 = (M + (2 * m) + (2 * I_wheel / (r**2)))
    term3 = (M * current_l)**2
    Qeq = (term1 * term2) - term3

    A23 = - ( (M**2) * (current_l**2) * g ) / Qeq   
    A43 = ( M * current_l * g * term2 ) / Qeq
            
    A = np.array([
        [0, 1, 0,   0, 0, 0],
        [0, 0, A23, 0, 0, 0],
        [0, 0, 0,   1, 0, 0],
        [0, 0, A43, 0, 0, 0],
        [0, 0, 0,   0, 0, 1],
        [0, 0, 0,   0, 0, 0]
    ])

    term4 = r * ( (m * d) + ((I_wheel * d)/(r**2)) + (2 * J_delta / d)  ) 
    B21 = ( J_p + (M * (current_l**2)) + (M * current_l * r) ) / (Qeq * r)
    B22 = B21
    B41 = -( ( (M * current_l) / r ) + term2 ) / Qeq
    B42 = B41
    B61 = -1 / term4
    B62 = 1 / term4

    B = np.array([
        [0,     0],
        [B21, B22],
        [0,     0],
        [B41, B42],
        [0,     0],
        [B61, B62]
    ])

    Q = np.diag([0.0, 30.0, 800.0, 1.0, 1.0, 1.0]) 
    R = np.array([[1.0, 0.0], [0.0, 1.0]]) 

    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    np.set_printoptions(precision=2, suppress=True)
    if(pos == "stand"):
        print("Matrix A in Stand is: \n", A)
        print("Matrix B in Stand is: \n", B)
        print("Value K in Stand is: \n", K)
    elif(pos == "squat"):
        print("Matrix A in Squat is: \n", A)
        print("Matrix B in Squat is: \n", B)
        print("Value K in Squat is: \n", K)
    else:
        print("Matrix A:\n", A)
        print("Matrix B:\n", B)
        print("Value K is: ", K)
        
    print("\n")
    return K

# ==========================================
# 2. Define pose
# ==========================================

posSquat = np.array([1.27, -2.127, 0, 1.27, -2.127, 0])

posStand = np.array([0.7, -1.5, 0, 0.7, -1.5, 0]) 

CGSquat = 0.23 
CGStand = 0.314

KSquat = LQR(CGSquat, pos="squat")
KStand = LQR(CGStand, pos="stand")
# ==========================================
# 3. Helper Functions
# ==========================================
def get_euler(quat):
    w, x, y, z = quat
    t2 = 2.0 * (w*y - z*x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w*z + x*y)
    t4 = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(t3, t4)
    return pitch, yaw

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# ==========================================
# 4. Main Simulation
# ==========================================
modelPath = './crazydog_urdf/urdf/scene.xml'
model = mujoco.MjModel.from_xml_path(modelPath)
data = mujoco.MjData(model)

# PD Gains
kps = np.array([150, 200 ,0, 150, 200, 0])
kds = np.array([30, 80, 0, 30, 80, 0])

# Logging
logTime = []           
logPitch = []          
logwheelSpeed = []     
logTorque = []
logYaw = []

# ==========================================
# Turning Control
# ==========================================
pressedKeys = set()
def Press(key):
    try:
        pressedKeys.add(key)
    except AttributeError:
        pass

def Release(key):
    if key in pressedKeys:
        pressedKeys.remove(key)

listener = keyboard.Listener(on_press=Press, on_release=Release)
listener.start()
# ==========================================


with mujoco.viewer.launch_passive(model, data) as viewer:

    target_body_name = "base_link" 
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body_id
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90

    lastTime = data.time
    currentVelocity = 0.0
    finalVelocity = 4.0
    Acceleration = 0.2

    torqueLimit = 15.0
    lastTorque = 0.0
    
    targetYaw = 0.0
    TurningSpeed = 0.3
    recoverySpeed = 5.0

    while viewer.is_running():
        step_start = time.time()

        turning = False
        if keyboard.Key.left in pressedKeys:
            targetYaw += TurningSpeed
            turning = True
        if keyboard.Key.right in pressedKeys:
            targetYaw -= TurningSpeed
            turning = True
        
        if not turning:
            if targetYaw > 0:
                targetYaw -= TurningSpeed * recoverySpeed
                if targetYaw < 0: targetYaw = 0
            elif targetYaw < 0:
                targetYaw += TurningSpeed * recoverySpeed
                if targetYaw > 0: targetYaw = 0 

        targetYaw = np.clip(targetYaw, -50, 50)
        target_yaw_rad = np.radians(targetYaw)

        # Read Sensor data
        quat = data.sensordata[18:22]
        gyro = data.sensordata[22:25]
        gyroY = gyro[1]       
        gyroZ = gyro[2]       

        leftWheelVelocity = data.sensordata[8]
        rightWheelVelocity = data.sensordata[11]

        pitch, yaw = get_euler(quat)
        averageWheelVelocity = (leftWheelVelocity + rightWheelVelocity) / 2.0

        currentTime = data.time
        dt = currentTime - lastTime
        lastTime = currentTime

        start_run_time = 6.0
        if currentTime < start_run_time:
            currentVelocity = 0.15
            
        else:
            if currentVelocity < finalVelocity:
                currentVelocity += Acceleration * dt
            
            elif currentVelocity > finalVelocity:
                currentVelocity = finalVelocity

            if turning:
                currentVelocity -= Acceleration * dt
                if currentVelocity < 0: currentVelocity = 0

        ratio = np.clip(abs(currentVelocity) / finalVelocity, 0.0, 1.0)
        KCurrent = (1 - ratio) * KSquat + ratio * KStand
        current_target_dof = (1 - ratio) * posSquat + ratio * posStand

        yaw_error = yaw - target_yaw_rad

        state = np.array([
            0.0,
            (averageWheelVelocity * r) - currentVelocity,
            pitch,
            gyroY,
            yaw_error,
            gyroZ
        ])

        Torque_LQR = -KCurrent @ state
        Torque_LQR[0] = np.clip(Torque_LQR[0], -15, 15)
        Torque_LQR[1] = np.clip(Torque_LQR[1], -15, 15)
        lastTorque = ( Torque_LQR[0] + Torque_LQR[1] ) / 2.0
        
        tau = pd_control(current_target_dof, data.sensordata[:6], kps, np.zeros(6), data.sensordata[6:12], kds)

        Stable_time = 0.3
        if currentTime < Stable_time:
            tau[2] = 0 
            tau[5] = 0
        else:
            tau[2] = Torque_LQR[0]
            tau[5] = Torque_LQR[1]

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Logging
        logTime.append(data.time)
        logPitch.append(np.degrees(pitch))
        logwheelSpeed.append(averageWheelVelocity * r)
        logTorque.append(Torque_LQR[0]) 
        logYaw.append(np.degrees(yaw_error))

        # Loop timing
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

listener.stop()

# ==========================================
# 5. Plotting
# ==========================================
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(logTime, logPitch, label='Pitch (deg)')
# plt.axhline(pitchLimit, color='r', linestyle='--')
plt.title('Body Pitch Angle')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(logTime, logwheelSpeed, label='Velocity (m/s)', color='orange')
plt.axhline(finalVelocity, color='r', linestyle='--', label='Target')
plt.ylim(0, finalVelocity + 1)
plt.title('Wheels Velocity')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(logTime, logTorque, label='Torque (Nm)', color='green')
plt.title('Wheels Torque')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(logTime, logYaw, label='Yaw Error (deg)', color='purple')
plt.title('Body Yaw Angle')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()