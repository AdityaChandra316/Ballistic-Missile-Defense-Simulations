import numpy as np
import math  
import random
from trajectory import calculate_impact_point, calculate_closest_distance # , Multithreaded_Distance_Calculator
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R = 6371000.0
M = 5.972168e24
EARTH_ANGULAR_VELOCITY = 0.00007292115
G = 6.6743015e-11 
g = G * M / (R * R)
dt = 1.0 / 256.0
earth_orientation_matrix = None
simulation_time = 0.0

def normalise(vector):
  mag = np.linalg.norm(vector)
  if mag == 0.0:
    return np.array([0.0, 0.0, 0.0])
  return vector / mag

def grav_field(point):
  r = np.linalg.norm(point)
  return -G * M * point / (r * r * r)

def lat_lng_to_point(coords, h):
  lat = coords[0]
  lng = coords[1]
  x = math.cos(lat) * math.cos(lng) * h
  y = math.cos(lat) * math.sin(lng) * h
  z = math.sin(lat) * h
  return np.array([x, y, z])

def point_to_lat_lng(point):
  lat = math.atan2(point[2], math.sqrt(point[0] ** 2 + point[1] ** 2))
  lng = math.atan2(point[1], point[0])
  return (lat, lng)  

def icbm_acc_profile(time_since_launch):
  return 4.0 * g
  """
  payload = 680.0
  if time_since_launch <= 60.0:
    burn_rate = (23077.0 - 2292.0) / 60.0
    return 792000.0 / (33709.0 + payload - burn_rate * time_since_launch)
  elif time_since_launch <= 126.0:
    time_since_stage_sep = time_since_launch - 60.0
    burn_rate = (7032.0 - 795.0) / 66.0
    return 267700.0 / (10632.0 + payload - burn_rate * time_since_stage_sep)
  elif time_since_launch <= 187.0:
    time_since_stage_sep = time_since_launch - 126.0
    burn_rate = (3600.0 - 400.0) / 61.0
    return 152000.0 / (3600.0 + payload - burn_rate * time_since_stage_sep)
  return 0.0
  """

def abm_acc_profile(time_since_launch):
  return 4.0 * g
 
def distance(coords_1, coords_2):
  lat_1 = coords_1[0]
  lng_1 = coords_1[1]
  lat_2 = coords_2[0]
  lng_2 = coords_2[1]
  d_lat = lat_2 - lat_1
  d_lng = lng_2 - lng_1
  return 2.0 * R * math.asin(math.sqrt((1.0 - math.cos(d_lat) + math.cos(lat_2) * math.cos(lat_1) * (1.0 - math.cos(d_lng))) / 2.0))

def make_orientation_matrix(time):
  rotation_angle = EARTH_ANGULAR_VELOCITY * time
  sin_rotation_angle = math.sin(rotation_angle)
  cos_rotation_angle = math.cos(rotation_angle)
  return np.array([
    [cos_rotation_angle, sin_rotation_angle, 0.0],
    [-sin_rotation_angle, cos_rotation_angle, 0.0],
    [0.0, 0.0, 1.0]
  ])

def calculate_impact_error(position, velocity):
  impact_data = calculate_impact_point(position, velocity)
  impact_point = impact_data['impact_point']
  time_to_impact = impact_data['time_to_impact']
  impact_coords = point_to_lat_lng(make_orientation_matrix(-(simulation_time + time_to_impact)).dot(impact_point))
  return distance(impact_coords, target_coords)

class Rocket:
  grounded = True
  position = None
  velocity = None
  launch_position =  None
  thrust_direction = None
  time_since_launch = 0.0
  acc_profile = None
  impact_point = None
  impact_velocity =  None
  time_to_impact = None
  engines_on = False
  def __init__(self, coords, acc_profile):
    self.launch_position = lat_lng_to_point(coords, R)
    self.position = self.launch_position
    self.acc_profile = acc_profile
  def launch(self):
    self.velocity = np.cross(np.array([0.0, 0.0, EARTH_ANGULAR_VELOCITY]), self.position)
    self.grounded = False
    self.engines_on = True
  def update(self, thrust_direction):
    if self.grounded:
      self.position = earth_orientation_matrix.dot(self.launch_position)
      return
    thrust_acc = np.array([0.0, 0.0, 0.0])
    if self.engines_on:
      thrust_acc = self.acc_profile(self.time_since_launch) * thrust_direction
    acc = thrust_acc + grav_field(self.position)
    self.position += self.velocity * dt + 0.5 * acc * dt * dt
    self.velocity += acc * dt
    self.time_since_launch += dt

launch_coords = (0.6229, 0.8976)
icbm = Rocket(launch_coords, icbm_acc_profile)
icbm.launch()
target_coords = (0.5943, -2.0637)
target_point = lat_lng_to_point(target_coords, R)
abm_coords = (1.0687, -2.6169)
abm = Rocket(abm_coords, abm_acc_profile)

def icbm_obj_fun(vel):
  vel_scaled = np.array(vel) * 1e3
  return calculate_impact_error(icbm.position, vel_scaled) / 1e6

def desired_icbm_velocity(init_vel, init_radius, num_iter):
  num_iter_minus_one = num_iter - 1
  velocity = np.array(init_vel) * 1e-3
  radius = init_radius
  current_error = None
  for i in range(num_iter):
    m_velocity = np.array(velocity)
    err_old = icbm_obj_fun(velocity)
    x = []
    y = []
    for j in range(30):
      rand = np.random.randn(3)
      rand *= radius * (np.random.uniform() ** (1.0 / 3.0)) / np.linalg.norm(rand)
      overall_vel = m_velocity + rand
      x.append(rand)
      y.append(icbm_obj_fun(overall_vel) - err_old)
    jacobian = np.linalg.lstsq(x, y, rcond=None)[0]
    jacobian_mag = np.linalg.norm(jacobian)
    jacobian_dir = jacobian
    if jacobian_mag != 0.0:
      jacobian_dir *= radius / jacobian_mag
    velocity -= jacobian
    err_pred = jacobian.dot(np.array([
      velocity[0] - m_velocity[0],
      velocity[1] - m_velocity[1],
      velocity[2] - m_velocity[2]
    ])) + err_old
    err_real = icbm_obj_fun(velocity)
    d_pred = err_pred - err_old
    d_real = err_real - err_old
    k = d_real / (d_pred + 1e-15)
    if k < 0.1:
      radius *= 0.5
      velocity = m_velocity
      if i == num_iter_minus_one:
        current_error = err_old 
    else:
      radius *= 1.1
      if i == num_iter_minus_one:
        current_error = err_real
  return {
    'desired_velocity': velocity * 1e3,
    'error': current_error * 1e6,
  }

desired_icbm_velocity_data = desired_icbm_velocity(icbm.velocity, 0.25, 100)
previous_impact_error = None

def icbm_desired_thrust_vector():
  global desired_icbm_velocity_data
  old_desired_velocity = desired_icbm_velocity_data['desired_velocity'] 
  desired_icbm_velocity_data = desired_icbm_velocity(
    old_desired_velocity,
    2 ** -12,
    1
  )
  v_tbg = desired_icbm_velocity_data['desired_velocity'] - icbm.velocity
  thrust_vector = normalise(v_tbg)
  return thrust_vector

def check_for_abm_launch():
  if abm.grounded == False:
    return
  abm_position = abm.position
  direction_to_icbm = icbm.position - abm_position
  if icbm.engines_on == False:
    print(simulation_time, abm_position.dot(direction_to_icbm))
  if abm_position.dot(direction_to_icbm) > 0.0:
    abm.launch()

def abm_obj_fun(vel):
  vel_scaled = np.array(vel) * 1e3
  d_position = abm.position - icbm.position
  dist_rate = (d_position.dot(vel_scaled - icbm.velocity) / np.linalg.norm(d_position))
  closest_distance = calculate_closest_distance(abm.position, vel_scaled, icbm.position, icbm.velocity)
  return 1e-6 * closest_distance + max(1e-3 * dist_rate, 0.0)

def desired_abm_velocity(init_vel, init_radius, num_iter):
  num_iter_minus_one = num_iter - 1
  velocity = np.array(init_vel) * 1e-3
  radius = init_radius
  current_error = None
  for i in range(num_iter):
    m_velocity = np.array(velocity)
    err_old = abm_obj_fun(velocity)
    x = []
    y = []
    for j in range(30):
      rand = np.random.randn(3)
      rand *= radius * (np.random.uniform() ** (1.0 / 3.0)) / np.linalg.norm(rand)
      rand_x = rand[0]
      rand_y = rand[1]
      rand_z = rand[2]
      x.append([
        rand_x,
        rand_y,
        rand_z
      ])
      y.append(abm_obj_fun(m_velocity + rand) - err_old)
    sol = np.linalg.lstsq(x, y, rcond=None)
    jacobian = sol[0]
    jacobian_mag = np.linalg.norm(jacobian)
    jacobian_dir = jacobian
    if jacobian_mag != 0.0:
      jacobian_dir *= radius / jacobian_mag
    velocity -= jacobian_dir
    v_x = velocity[0] - m_velocity[0]
    v_y = velocity[1] - m_velocity[1]
    v_z = velocity[2] - m_velocity[2]
    err_pred = jacobian.dot(np.array([
      v_x,
      v_y,
      v_z,
    ])) + err_old
    err_real = abm_obj_fun(velocity)
    d_pred = err_pred - err_old
    d_real = err_real - err_old
    k = d_real / (d_pred + 1e-15)
    if k < 0.1:
      radius *= 0.5
      velocity = m_velocity
      current_error = err_old 
    else:
      radius *= 1.1
      current_error = err_real
  velocity *= 1e3
  return {
    'desired_velocity': velocity,
    'error': calculate_closest_distance(abm.position, velocity, icbm.position, icbm.velocity),
  }

desired_abm_velocity_data = None
prev_closest_distance = None
is_terminal_phase = False

def abm_desired_thrust_vector():
  global desired_abm_velocity_data
  old_desired_velocity = desired_abm_velocity_data['desired_velocity'] 
  desired_abm_velocity_data = desired_abm_velocity(
    old_desired_velocity,
    2 ** -12,
    1
  )
  v_tbg = desired_abm_velocity_data['desired_velocity'] - abm.velocity
  grav_acc = grav_field(abm.position)
  abm_acc = abm_acc_profile(abm.time_since_launch)
  a = np.linalg.norm(v_tbg) ** 2.0
  b = -2.0 * v_tbg.dot(grav_acc)
  c = np.linalg.norm(grav_acc) ** 2.0 - abm_acc ** 2.0
  k = (-b + math.sqrt(b ** 2.0 - 4.0 * a * c)) / (2.0 * a)
  thrust_vector = (k * v_tbg - grav_acc) / abm_acc
  return thrust_vector

intercept_altitude = 2400000.0 / math.sqrt(2.0)
c = 2400000.0 # ABM distance

target_pos = lat_lng_to_point(target_coords, R)

def get_abm_positions():
  icbm_position = icbm.position
  a = R
  b = np.linalg.norm(icbm_position)
  radial_out = normalise(icbm_position)
  icbm_velocity = icbm.velocity
  forward_vel = normalise(icbm_velocity - radial_out * radial_out.dot(icbm_velocity))
  right_vel = np.cross(radial_out, forward_vel)
  positions = []
  theta = math.acos((c ** 2.0 - a ** 2.0 - b ** 2.0) / (-2.0 * a * b))
  mag = math.fabs(math.tan(theta) * b)
  for angle in range(0, 198, 18):
    rad = angle * math.pi / 180.0
    vel_dir = math.cos(rad) * forward_vel + math.sin(rad) * right_vel
    positions.append(normalise(vel_dir * mag + icbm_position) * R)
  return positions

f = open("output.txt", "w")

prev_dist_between_abm_icbm = None

abm_line = []
icbm_line = []

while True:
  earth_orientation_matrix = make_orientation_matrix(simulation_time)
  inv_earth_orientation_matrix = make_orientation_matrix(-simulation_time)

  check_for_abm_launch()
  icbm_thrust_vector = np.array([0.0, 0.0, 0.0])
  radial_out = normalise(icbm.position)
  icbm_altitude = np.linalg.norm(icbm.position) - R
  
  icbm_line.append(inv_earth_orientation_matrix.dot(icbm.position))

  if icbm.engines_on == True:
    icbm_thrust_vector = icbm_desired_thrust_vector()
    current_impact_error = calculate_impact_error(icbm.position, icbm.velocity)

    print(
      simulation_time, 
      icbm_altitude,
      current_impact_error
    )

    if not previous_impact_error is None and current_impact_error > previous_impact_error and current_impact_error < 1000.0:
      icbm.engines_on = False
    
    previous_impact_error = current_impact_error

  abm_thrust_vector = np.array([0.0, 0.0, 0.0])
    
  dist_between_abm_icbm = np.linalg.norm(icbm.position - abm.position)

  if abm.grounded == False:
    abm_line.append(inv_earth_orientation_matrix.dot(abm.position))

    closest_distance = calculate_closest_distance(abm.position, abm.velocity, icbm.position, icbm.velocity)

    if not is_terminal_phase:

      if desired_abm_velocity_data is None:
        desired_abm_velocity_data = desired_abm_velocity(abm.velocity, 0.25, 100)
        print("Delta V", np.linalg.norm(desired_abm_velocity_data['desired_velocity'] - abm.velocity))
        print("Error", desired_abm_velocity_data['error'])
        time.sleep(10.0)

      if not prev_closest_distance is None and closest_distance > prev_closest_distance and closest_distance < 1000.0:
        abm.engines_on = False

      if abm.engines_on == False and dist_between_abm_icbm < 70000.0:    
        desired_abm_velocity_data = desired_abm_velocity(abm.velocity, 0.25, 10)
        print("Delta V", np.linalg.norm(desired_abm_velocity_data['desired_velocity'] - abm.velocity))
        print("Error", desired_abm_velocity_data['error'])
        time.sleep(10.0)
        abm.engines_on = True
        is_terminal_phase = True
      
    prev_closest_distance = closest_distance
    
    if abm.engines_on:
      desired_abm_thrust_vector = abm_desired_thrust_vector()
      abm_thrust_vector = desired_abm_thrust_vector

    print(
      abm.time_since_launch, 
      np.linalg.norm(abm.position) - R,
      dist_between_abm_icbm,
      closest_distance    
    )
    
    if not prev_dist_between_abm_icbm is None and dist_between_abm_icbm > prev_dist_between_abm_icbm and dist_between_abm_icbm < 1000.0:
      break

    prev_dist_between_abm_icbm = dist_between_abm_icbm
    
  icbm.update(icbm_thrust_vector)
  abm.update(abm_thrust_vector)

  """
  if icbm.velocity.dot(radial_out) < 0.0 and icbm_altitude < intercept_altitude:
    abm_positions = get_abm_positions()

    # print(icbm.velocity.dot(normalise(icbm.position)))

    for abm_position in abm_positions:
      abm.position = abm_position
      abm.velocity = np.cross(np.array([0.0, 0.0, EARTH_ANGULAR_VELOCITY]), abm_position)

      for rep in range(10):
        # print(abm.velocity)
        desired_abm_velocity_data = desired_abm_velocity(abm.velocity, 0.25, 100)
        f.write(str(np.linalg.norm(desired_abm_velocity_data['desired_velocity'] - abm.velocity)))
        print(desired_abm_velocity_data['error'])

        if rep != 9:
          f.write(",")

      f.write("\n")

    break
  """
  
  simulation_time += dt

f.close()

abm_line = np.array(abm_line) / 1e6
icbm_line = np.array(icbm_line) / 1e6

step = 4
img = mpimg.imread("Blue_Marble_2002.png")
img = img[..., :3]  # drop alpha if present
img = img[::step, ::step]
img = img.astype(np.float32)
if img.max() > 1.0:  # normalize if uint8 0..255
    img /= 255.0

# --- make a sphere mesh whose resolution matches the image ---
H, W = img.shape[:2]
lon = np.linspace(-np.pi, np.pi, W)          # -180..180
lat = np.linspace( np.pi/2, -np.pi/2, H)    #  90..-90 (top->bottom)
lon2d, lat2d = np.meshgrid(lon, lat)

X = R * np.cos(lat2d) * np.cos(lon2d) / 1e6
Y = R * np.cos(lat2d) * np.sin(lon2d) / 1e6
Z = R * np.sin(lat2d) / 1e6

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # fill the window
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()

ax.set_xlim(-R / 1e6, R / 1e6)
ax.set_ylim(-R / 1e6, R / 1e6)
ax.set_zlim(-R / 1e6, R / 1e6)

ax.computed_zorder = False   # <-- key line (manual z-order)

# Texture-map the Earth
ax.plot_surface(X, Y, Z, facecolors=img, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, zorder=0)

k = 256 # keep every kth point
ax.plot(abm_line[::k, 0], abm_line[::k, 1], abm_line[::k, 2], label="ABM Trajectory", zorder=10)
ax.plot(icbm_line[::k, 0], icbm_line[::k, 1], icbm_line[::k, 2], label="ICBM Trajectory", zorder=10)
ax.set_box_aspect([1, 1, 1])
ax.set_axis_off()
ax.legend()

ax.view_init(elev=45.0, azim=135.0)
plt.show()
