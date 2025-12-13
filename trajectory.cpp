#include <iostream>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <thread>
#include <vector>
#include <chrono>

float G = 6.6743015e-11f; 
float M = 5.972168e24f;
float R = 6371000.0f;
float time_limit = 1.0f / 256.0f;

void gravitational_field(float* point, float* field) {
  float r = sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
  float r_cubed = r * r * r;
  field[0] = -G * M * point[0] / r_cubed;
  field[1] = -G * M * point[1] / r_cubed;
  field[2] = -G * M * point[2] / r_cubed;
}

void print_point(float* point) {
  std::cout << point[0] << " " << point[1] << " " << point[2] << std::endl;
}

pybind11::dict calculate_impact_point(pybind11::array_t<float> position, pybind11::array_t<float> velocity) {
  float m_dt = 8.0f;
  float m_time = 0.0f;

  pybind11::buffer_info position_buf = position.request(); 
  pybind11::buffer_info velocity_buf = velocity.request();
  float *position_ptr = static_cast<float *>(position_buf.ptr);
  float *velocity_ptr = static_cast<float *>(velocity_buf.ptr);

  float m_position[3];
  float m_velocity[3];

  m_position[0] = position_ptr[0];
  m_position[1] = position_ptr[1];
  m_position[2] = position_ptr[2];

  m_velocity[0] = velocity_ptr[0];
  m_velocity[1] = velocity_ptr[1];
  m_velocity[2] = velocity_ptr[2];

  while (true) {
    m_time += m_dt;

    float acc_old[3];
    gravitational_field(m_position, acc_old);
    m_position[0] += m_velocity[0] * m_dt + 0.5f * acc_old[0] * m_dt * m_dt;
    m_position[1] += m_velocity[1] * m_dt + 0.5f * acc_old[1] * m_dt * m_dt;
    m_position[2] += m_velocity[2] * m_dt + 0.5f * acc_old[2] * m_dt * m_dt;
    float acc_new[3];
    gravitational_field(m_position, acc_new);
    m_velocity[0] += 0.5f * (acc_old[0] + acc_new[0]) * m_dt;
    m_velocity[1] += 0.5f * (acc_old[1] + acc_new[1]) * m_dt;
    m_velocity[2] += 0.5f * (acc_old[2] + acc_new[2]) * m_dt;

    float altitude = sqrt(m_position[0] * m_position[0] + m_position[1] * m_position[1] + m_position[2] * m_position[2]) - R;

    if (altitude < 0.0f) {
      float lower_time = -m_dt;
      float upper_time = 0.0f;
      float middle_time = 0.5f * (lower_time + upper_time);
      float previous_m_point = 0.0f;
      
      while (true) {
        if (upper_time - lower_time <= time_limit) {
          pybind11::array_t<float> result = pybind11::array_t<float>(3);
          pybind11::buffer_info result_buffer = result.request();
          float *result_ptr = static_cast<float *>(result_buffer.ptr);
          
          result_ptr[0] = m_position[0];
          result_ptr[1] = m_position[1];
          result_ptr[2] = m_position[2];
          
          pybind11::dict output;
          output["impact_point"] = result;
          output["time_to_impact"] = m_time + previous_m_point;
          return output;
        }

        float best_dt = middle_time - previous_m_point;
        float sign = best_dt < 0.0f ? -1.0f : 1.0f;

        float acc_old[3];
        gravitational_field(m_position, acc_old);
        m_position[0] += m_velocity[0] * best_dt + 0.5f * acc_old[0] * best_dt * best_dt * sign;
        m_position[1] += m_velocity[1] * best_dt + 0.5f * acc_old[1] * best_dt * best_dt * sign;
        m_position[2] += m_velocity[2] * best_dt + 0.5f * acc_old[2] * best_dt * best_dt * sign;
        float acc_new[3]; 
        gravitational_field(m_position, acc_new);
        m_velocity[0] += 0.5f * (acc_old[0] + acc_new[0]) * best_dt;
        m_velocity[1] += 0.5f * (acc_old[1] + acc_new[1]) * best_dt;
        m_velocity[2] += 0.5f * (acc_old[2] + acc_new[2]) * best_dt;
        
        float impact_altitude = sqrt(m_position[0] * m_position[0] + m_position[1] * m_position[1] + m_position[2] * m_position[2]) - R;
        
        if (impact_altitude < 0.0f) {
          upper_time = middle_time;
        } else {
          lower_time = middle_time;
        }

        previous_m_point = middle_time;
        middle_time = 0.5f * (lower_time + upper_time);
      }
    }
  }
}

float calculate_closest_distance_cpp(float* m_abm_position, float* m_abm_velocity, float* m_icbm_position, float* m_icbm_velocity) {
  float m_dt = 8.0f;
  float m_time = 0.0f;

  while (true) {
    bool is_breaking = false;

    m_time += m_dt;

    float acc_abm_old[3];
    gravitational_field(m_abm_position, acc_abm_old);
    m_abm_position[0] += m_abm_velocity[0] * m_dt + 0.5f * acc_abm_old[0] * m_dt * m_dt;
    m_abm_position[1] += m_abm_velocity[1] * m_dt + 0.5f * acc_abm_old[1] * m_dt * m_dt;
    m_abm_position[2] += m_abm_velocity[2] * m_dt + 0.5f * acc_abm_old[2] * m_dt * m_dt;
    float acc_abm_new[3];
    gravitational_field(m_abm_position, acc_abm_new);
    m_abm_velocity[0] += 0.5f * (acc_abm_old[0] + acc_abm_new[0]) * m_dt;
    m_abm_velocity[1] += 0.5f * (acc_abm_old[1] + acc_abm_new[1]) * m_dt;
    m_abm_velocity[2] += 0.5f * (acc_abm_old[2] + acc_abm_new[2]) * m_dt;

    float acc_icbm_old[3];
    gravitational_field(m_icbm_position, acc_icbm_old);
    m_icbm_position[0] += m_icbm_velocity[0] * m_dt + 0.5f * acc_icbm_old[0] * m_dt * m_dt;
    m_icbm_position[1] += m_icbm_velocity[1] * m_dt + 0.5f * acc_icbm_old[1] * m_dt * m_dt;
    m_icbm_position[2] += m_icbm_velocity[2] * m_dt + 0.5f * acc_icbm_old[2] * m_dt * m_dt;
    float acc_icbm_new[3];
    gravitational_field(m_icbm_position, acc_icbm_new);
    m_icbm_velocity[0] += 0.5f * (acc_icbm_old[0] + acc_icbm_new[0]) * m_dt;
    m_icbm_velocity[1] += 0.5f * (acc_icbm_old[1] + acc_icbm_new[1]) * m_dt;
    m_icbm_velocity[2] += 0.5f * (acc_icbm_old[2] + acc_icbm_new[2]) * m_dt;

    float distance_rate = (
      (m_abm_position[0] - m_icbm_position[0]) * (m_abm_velocity[0] - m_icbm_velocity[0]) + 
      (m_abm_position[1] - m_icbm_position[1]) * (m_abm_velocity[1] - m_icbm_velocity[1]) + 
      (m_abm_position[2] - m_icbm_position[2]) * (m_abm_velocity[2] - m_icbm_velocity[2])
    );

    float abm_altitude = sqrtf(m_abm_position[0] * m_abm_position[0] + m_abm_position[1] * m_abm_position[1] + m_abm_position[2] * m_abm_position[2]) - R;
    float icbm_altitude = sqrtf(m_icbm_position[0] * m_icbm_position[0] + m_icbm_position[1] * m_icbm_position[1] + m_icbm_position[2] * m_icbm_position[2]) - R;

    if (distance_rate > 0.0f || abm_altitude < 0.0f || icbm_altitude < 0.0f) {
      float lower_time = -m_dt;
      float upper_time = 0.0f;
      float middle_time = 0.5f * (lower_time + upper_time);
      float previous_m_point = 0.0;

      while (true) {
        if (upper_time - lower_time <= time_limit) {
          is_breaking = true;
          break;
        }

        float best_dt = middle_time - previous_m_point;
        float sign = best_dt < 0.0f ? -1.0f : 1.0f;

        float acc_abm_old[3];
        gravitational_field(m_abm_position, acc_abm_old);
        m_abm_position[0] += m_abm_velocity[0] * best_dt + 0.5f * acc_abm_old[0] * best_dt * best_dt * sign;
        m_abm_position[1] += m_abm_velocity[1] * best_dt + 0.5f * acc_abm_old[1] * best_dt * best_dt * sign;
        m_abm_position[2] += m_abm_velocity[2] * best_dt + 0.5f * acc_abm_old[2] * best_dt * best_dt * sign;
        float acc_abm_new[3];
        gravitational_field(m_abm_position, acc_abm_new);
        m_abm_velocity[0] += 0.5f * (acc_abm_old[0] + acc_abm_new[0]) * best_dt;
        m_abm_velocity[1] += 0.5f * (acc_abm_old[1] + acc_abm_new[1]) * best_dt;
        m_abm_velocity[2] += 0.5f * (acc_abm_old[2] + acc_abm_new[2]) * best_dt;

        float acc_icbm_old[3];
        gravitational_field(m_icbm_position, acc_icbm_old);
        m_icbm_position[0] += m_icbm_velocity[0] * best_dt + 0.5f * acc_icbm_old[0] * best_dt * best_dt * sign;
        m_icbm_position[1] += m_icbm_velocity[1] * best_dt + 0.5f * acc_icbm_old[1] * best_dt * best_dt * sign;
        m_icbm_position[2] += m_icbm_velocity[2] * best_dt + 0.5f * acc_icbm_old[2] * best_dt * best_dt * sign;
        float acc_icbm_new[3];
        gravitational_field(m_icbm_position, acc_icbm_new);
        m_icbm_velocity[0] += 0.5f * (acc_icbm_old[0] + acc_icbm_new[0]) * best_dt;
        m_icbm_velocity[1] += 0.5f * (acc_icbm_old[1] + acc_icbm_new[1]) * best_dt;
        m_icbm_velocity[2] += 0.5f * (acc_icbm_old[2] + acc_icbm_new[2]) * best_dt;

        float inner_abm_altitude = sqrtf(m_abm_position[0] * m_abm_position[0] + m_abm_position[1] * m_abm_position[1] + m_abm_position[2] * m_abm_position[2]) - R;
        float inner_icbm_altitude = sqrtf(m_icbm_position[0] * m_icbm_position[0] + m_icbm_position[1] * m_icbm_position[1] + m_icbm_position[2] * m_icbm_position[2]) - R;

        float inner_distance_rate = (
          (m_abm_position[0] - m_icbm_position[0]) * (m_abm_velocity[0] - m_icbm_velocity[0]) + 
          (m_abm_position[1] - m_icbm_position[1]) * (m_abm_velocity[1] - m_icbm_velocity[1]) + 
          (m_abm_position[2] - m_icbm_position[2]) * (m_abm_velocity[2] - m_icbm_velocity[2])
        );

        if (inner_distance_rate > 0.0f || inner_abm_altitude < 0.0f || inner_icbm_altitude < 0.0f) {
          upper_time = middle_time;
        } else {
          lower_time = middle_time;
        }

        previous_m_point = middle_time;
        middle_time = 0.5f * (lower_time + upper_time);
      }
    }

    if (is_breaking) break;
  }
  return sqrtf(powf(m_abm_position[0] - m_icbm_position[0], 2.0f) + powf(m_abm_position[1] - m_icbm_position[1], 2.0f) + powf(m_abm_position[2] - m_icbm_position[2], 2.0f));
}

float calculate_closest_distance(pybind11::array_t<float> abm_position, pybind11::array_t<float> abm_velocity, pybind11::array_t<float> icbm_position, pybind11::array_t<float> icbm_velocity) {
  float m_time = 0.0f;

  pybind11::buffer_info abm_position_buf = abm_position.request(); 
  pybind11::buffer_info abm_velocity_buf = abm_velocity.request();
  float *abm_position_ptr = static_cast<float *>(abm_position_buf.ptr);
  float *abm_velocity_ptr = static_cast<float *>(abm_velocity_buf.ptr);

  pybind11::buffer_info icbm_position_buf = icbm_position.request(); 
  pybind11::buffer_info icbm_velocity_buf = icbm_velocity.request();
  float *icbm_position_ptr = static_cast<float *>(icbm_position_buf.ptr);
  float *icbm_velocity_ptr = static_cast<float *>(icbm_velocity_buf.ptr);

  float m_abm_position[3] = {abm_position_ptr[0], abm_position_ptr[1], abm_position_ptr[2]};
  float m_abm_velocity[3] = {abm_velocity_ptr[0], abm_velocity_ptr[1], abm_velocity_ptr[2]};
  float m_icbm_position[3] = {icbm_position_ptr[0], icbm_position_ptr[1], icbm_position_ptr[2]};
  float m_icbm_velocity[3] = {icbm_velocity_ptr[0], icbm_velocity_ptr[1], icbm_velocity_ptr[2]};

  return calculate_closest_distance_cpp(m_abm_position, m_abm_velocity, m_icbm_position, m_icbm_velocity);
}

PYBIND11_MODULE(trajectory, handle) {
  handle.doc() = "Find the impact point given a position and velocity";
  handle.def("calculate_impact_point", &calculate_impact_point);
  handle.def("calculate_closest_distance", &calculate_closest_distance);
}
