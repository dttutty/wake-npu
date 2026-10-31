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

extern "C" void attention_row0_aie2(bfloat16 *restrict input,
                                      bfloat16 *restrict output) {
  const aie::rounding_mode old_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  const int valid_keys = static_cast<int>(static_cast<float>(input[0]));
  const bfloat16 *query = input + 2;
  const bfloat16 *keys = query + 64;
  const bfloat16 *values = keys + 32 * 64;

  float scores[32];
  float maximum = -1.0e30f;
  for (int key = 0; key < valid_keys; ++key) {
    float score = 0.0f;
    for (int dim = 0; dim < 64; ++dim)
      score += static_cast<float>(query[dim]) *
               static_cast<float>(keys[key * 64 + dim]);
    scores[key] = static_cast<float>(static_cast<bfloat16>(score));
    if (scores[key] > maximum)
      maximum = scores[key];
  }

  float probabilities[32];
  float sum = 0.0f;
  for (int key = 0; key < valid_keys; ++key) {
    probabilities[key] = exp_nonpositive(scores[key] - maximum);
    sum += probabilities[key];
  }
  for (int key = 0; key < valid_keys; ++key)
    probabilities[key] = static_cast<float>(
        static_cast<bfloat16>(probabilities[key] / sum));

  for (int dim = 0; dim < 64; ++dim) {
    float result = 0.0f;
    for (int key = 0; key < valid_keys; ++key)
      result += probabilities[key] *
                static_cast<float>(values[key * 64 + dim]);
    output[dim] = static_cast<bfloat16>(result);
  }
  aie::set_rounding(old_rounding);
}
