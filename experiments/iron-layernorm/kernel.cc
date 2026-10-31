#define NOCPP

#include <aie_api/aie.hpp>
#include <vec_math.h>

static void normalize_rows(bfloat16 *restrict input,
                           bfloat16 *restrict residual,
                           bfloat16 *restrict output) {
  constexpr int rows = 64;
  constexpr int cols = 32;
  constexpr float epsilon = 1e-5f;
  const aie::rounding_mode old_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);

  for (int row = 0; row < rows; ++row) {
    const int start = row * cols;
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      const int index = start + col;
      float value = static_cast<float>(input[index]);
      if (residual)
        value += static_cast<float>(residual[index]);
      sum += value;
    }
    const float mean = sum / cols;

    float squared_deviation = 0.0f;
    for (int col = 0; col < cols; ++col) {
      const int index = start + col;
      float value = static_cast<float>(input[index]);
      if (residual)
        value += static_cast<float>(residual[index]);
      const float delta = value - mean;
      squared_deviation += delta * delta;
    }
    const float inv_std = invsqrt(squared_deviation / cols + epsilon);

    for (int col = 0; col < cols; ++col) {
      const int index = start + col;
      float value = static_cast<float>(input[index]);
      if (residual)
        value += static_cast<float>(residual[index]);
      output[index] = static_cast<bfloat16>((value - mean) * inv_std);
    }
  }

  aie::set_rounding(old_rounding);
}

extern "C" void layer_norm_aie2(bfloat16 *restrict input,
                                  bfloat16 *restrict output) {
  normalize_rows(input, nullptr, output);
}

extern "C" void add_layer_norm_aie2(bfloat16 *restrict input,
                                      bfloat16 *restrict residual,
                                      bfloat16 *restrict output) {
  normalize_rows(input, residual, output);
}
