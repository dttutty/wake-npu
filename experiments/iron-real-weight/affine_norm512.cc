#define NOCPP

#include <aie_api/aie.hpp>
#include <vec_math.h>

extern "C" void add_affine_norm512_aie2(bfloat16 *restrict input_and_residual,
                                          float *restrict gamma_beta,
                                          bfloat16 *restrict output) {
  constexpr int cols = 512;
  constexpr float epsilon = 1.0e-5f;
  const aie::rounding_mode old_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);

  float sum = 0.0f;
  for (int col = 0; col < cols; ++col)
    sum += static_cast<float>(input_and_residual[col]) +
           static_cast<float>(input_and_residual[cols + col]);
  const float mean = sum / cols;

  float squared_deviation = 0.0f;
  for (int col = 0; col < cols; ++col) {
    const float value =
        static_cast<float>(input_and_residual[col]) +
        static_cast<float>(input_and_residual[cols + col]);
    const float delta = value - mean;
    squared_deviation += delta * delta;
  }
  const float inv_std = invsqrt(squared_deviation / cols + epsilon);

  for (int col = 0; col < cols; ++col) {
    const float value =
        static_cast<float>(input_and_residual[col]) +
        static_cast<float>(input_and_residual[cols + col]);
    const float normalized = (value - mean) * inv_std;
    output[col] = static_cast<bfloat16>(
        normalized * gamma_beta[col] + gamma_beta[cols + col]);
  }
  aie::set_rounding(old_rounding);
}
