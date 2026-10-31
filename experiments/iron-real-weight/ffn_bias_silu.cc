#define NOCPP

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <lut_based_ops.cpp>

extern "C" void ffn_bias_silu_aie2(float *restrict accumulator,
                                    float *restrict bias,
                                    bfloat16 *restrict output) {
  const aie::rounding_mode old_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  const auto half = aie::broadcast<bfloat16, 16>(0.5f);
  const auto one = aie::broadcast<bfloat16, 16>(1.0f);
  alignas(32) bfloat16 rounded[16];

  for (int col = 0; col < 2048; col += 16) {
    for (int lane = 0; lane < 16; ++lane)
      rounded[lane] = static_cast<bfloat16>(accumulator[col + lane] + bias[col + lane]);
    const auto x = aie::load_v<16>(rounded);
    aie::vector<bfloat16, 16> half_x = aie::mul(x, half);
    aie::vector<bfloat16, 16> tanh_half = getTanhBf16(half_x);
    auto tanh_plus_one = aie::add(tanh_half, one);
    aie::vector<bfloat16, 16> sigmoid = aie::mul(tanh_plus_one, half);
    aie::store_v(output + col, aie::mul(x, sigmoid).to_vector<bfloat16>());
  }
  aie::set_rounding(old_rounding);
}
