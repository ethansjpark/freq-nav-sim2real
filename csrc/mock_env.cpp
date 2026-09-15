#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

class MockPointNavEnv {
public:
  int image_size;
  int max_episode_steps;
  float success_distance;
  float step_size;
  float turn_angle_rad;
  float world_extent;

  int step_count = 0;
  float path_length = 0.0f;
  float pos_x = 0.0f, pos_y = 0.0f;
  float goal_x = 0.0f, goal_y = 0.0f;
  float heading_rad = 0.0f;
  float shortest_path = 0.0f;

  std::mt19937_64 rng;
  std::uniform_real_distribution<float> pos_dist;
  std::uniform_real_distribution<float> heading_dist;

  MockPointNavEnv(
      int image_size,
      int max_episode_steps,
      float success_distance,
      float step_size,
      float turn_angle_deg,
      float world_extent,
      uint64_t seed)
      : image_size(image_size),
        max_episode_steps(max_episode_steps),
        success_distance(success_distance),
        step_size(step_size),
        turn_angle_rad(turn_angle_deg * static_cast<float>(M_PI) / 180.0f),
        world_extent(world_extent),
        rng(seed),
        pos_dist(-world_extent, world_extent),
        heading_dist(static_cast<float>(-M_PI), static_cast<float>(M_PI)) {}

  float distance_to_goal() const {
    float dx = goal_x - pos_x;
    float dy = goal_y - pos_y;
    return std::sqrt(dx * dx + dy * dy);
  }

  py::array_t<uint8_t> render_obs() const {
    const int h = image_size;
    const int w = image_size;

    auto result = py::array_t<uint8_t>({h, w, 3});
    auto buf = result.mutable_unchecked<3>();

    float rel_x = goal_x - pos_x;
    float rel_y = goal_y - pos_y;
    float dist = std::max(1e-6f, std::sqrt(rel_x * rel_x + rel_y * rel_y));

    float rel_angle = std::atan2(rel_y, rel_x) - heading_rad;
    // Wrap to [-pi, pi].
    rel_angle = std::fmod(rel_angle + static_cast<float>(M_PI),
                          2.0f * static_cast<float>(M_PI));
    if (rel_angle < 0.0f) rel_angle += 2.0f * static_cast<float>(M_PI);
    rel_angle -= static_cast<float>(M_PI);

    int stripe_x = static_cast<int>(
        ((rel_angle / static_cast<float>(M_PI)) * 0.5f + 0.5f) *
        static_cast<float>(w - 1));
    float norm_dist = std::min(dist / (2.0f * world_extent), 1.0f);
    int stripe_y = static_cast<int>(norm_dist * static_cast<float>(h - 1));

    float proximity = std::max(0.0f, 1.0f - dist / (2.0f * world_extent));
    uint8_t prox_val = static_cast<uint8_t>(255.0f * proximity);

    // Fill image: channel 0 = proximity, channel 1 = distance stripe,
    //             channel 2 = bearing stripe.
    for (int r = 0; r < h; ++r) {
      for (int col = 0; col < w; ++col) {
        buf(r, col, 0) = prox_val;
        buf(r, col, 1) = 0;
        buf(r, col, 2) = 0;
      }
    }
    // Horizontal stripe at stripe_y (4px tall) on green channel.
    for (int r = stripe_y; r < std::min(stripe_y + 4, h); ++r) {
      for (int col = 0; col < w; ++col) {
        buf(r, col, 1) = 160;
      }
    }
    // Vertical stripe at stripe_x (4px wide) on blue channel.
    for (int r = 0; r < h; ++r) {
      for (int col = stripe_x; col < std::min(stripe_x + 4, w); ++col) {
        buf(r, col, 2) = 200;
      }
    }

    return result;
  }

  py::array_t<uint8_t> reset() {
    step_count = 0;
    path_length = 0.0f;
    pos_x = pos_dist(rng);
    pos_y = pos_dist(rng);
    goal_x = pos_dist(rng);
    goal_y = pos_dist(rng);
    heading_rad = heading_dist(rng);
    shortest_path = distance_to_goal();
    return render_obs();
  }

  py::tuple step(int action) {
    step_count++;
    float prev_x = pos_x, prev_y = pos_y;

    if (action == 1) {
      heading_rad += turn_angle_rad;
    } else if (action == 2) {
      heading_rad -= turn_angle_rad;
    } else if (action == 0) {
      pos_x += step_size * std::cos(heading_rad);
      pos_y += step_size * std::sin(heading_rad);
      pos_x = std::clamp(pos_x, -world_extent, world_extent);
      pos_y = std::clamp(pos_y, -world_extent, world_extent);
    } else {
      throw std::invalid_argument("Unsupported action: " + std::to_string(action));
    }

    float dx = pos_x - prev_x;
    float dy = pos_y - prev_y;
    path_length += std::sqrt(dx * dx + dy * dy);

    float dist = distance_to_goal();
    bool success = dist <= success_distance;
    bool done = success || step_count >= max_episode_steps;
    float reward = success ? 1.0f : 0.0f;

    py::dict info;
    info["success"] = static_cast<double>(success);
    info["path_length"] = static_cast<double>(path_length);
    info["shortest_path"] = static_cast<double>(shortest_path);
    info["distance_to_goal"] = static_cast<double>(dist);

    return py::make_tuple(render_obs(), reward, done, info);
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<MockPointNavEnv>(m, "MockPointNavEnv")
      .def(py::init<int, int, float, float, float, float, uint64_t>(),
           py::arg("image_size") = 224,
           py::arg("max_episode_steps") = 200,
           py::arg("success_distance") = 0.2f,
           py::arg("step_size") = 0.15f,
           py::arg("turn_angle_deg") = 15.0f,
           py::arg("world_extent") = 5.0f,
           py::arg("seed") = 0)
      .def("reset", &MockPointNavEnv::reset)
      .def("step", &MockPointNavEnv::step);
}
