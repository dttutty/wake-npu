#define NOCPP

#include <aie_api/aie.hpp>

static float exp_nonpositive(float value) {
  if (value < -20.0f)
    return 0.0f;
  float result = 1.0f + value / 16384.0f;
  for (int i = 0; i < 14; ++i)
    result *= result;
  return result;
}

extern "C" void masked_softmax64_aie2(bfloat16 *restrict input,
                                        bfloat16 *restrict output,
                                        const int32_t valid_keys) {
  const aie::rounding_mode old_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);

  for (int row = 0; row < 64; ++row) {
    const int base = row * 64;
    float maximum = -1.0e30f;
    for (int col = 0; col < valid_keys; ++col) {
      const float value = static_cast<float>(input[base + col]);
      if (value > maximum)
        maximum = value;
    }

    float exponentials[64];
    float sum = 0.0f;
    for (int col = 0; col < valid_keys; ++col) {
      const float value =
          exp_nonpositive(static_cast<float>(input[base + col]) - maximum);
      exponentials[col] = value;
      sum += value;
    }
    const float inverse_sum = valid_keys > 0 ? 1.0f / sum : 0.0f;
    for (int col = 0; col < 64; ++col) {
      const float value = col < valid_keys ? exponentials[col] * inverse_sum : 0.0f;
      output[base + col] = static_cast<bfloat16>(value);
    }
  }

  aie::set_rounding(old_rounding);
}
