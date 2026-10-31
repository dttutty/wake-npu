#define NOCPP

#include <aie_api/aie.hpp>

extern "C" void fused_vector(bfloat16 *restrict x, bfloat16 *restrict y,
                              bfloat16 *restrict z) {
  constexpr int count = 4096;
  constexpr int lanes = 64;
  const aie::vector<bfloat16, lanes> scale = aie::broadcast<bfloat16, lanes>(5.f);

  for (int i = 0; i < count; i += lanes) {
    const aie::vector<bfloat16, lanes> xv = aie::load_v<lanes>(x + i);
    const aie::vector<bfloat16, lanes> yv = aie::load_v<lanes>(y + i);
    const aie::accum<accfloat, lanes> product = aie::mul(xv, scale);
    const aie::accum<accfloat, lanes> sum = aie::add(product, yv);
    aie::store_v(z + i, sum.to_vector<bfloat16>());
  }
}
