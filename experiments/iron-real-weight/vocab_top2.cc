#define NOCPP

#include <aie_api/aie.hpp>

extern "C" void vocab_bias_top2_aie2(float *restrict logits,
                                       float *restrict bias,
                                       float *restrict top2) {
  float first = -1.0e30f;
  float second = -1.0e30f;
  int first_index = -1;
  int second_index = -1;
  for (int col = 0; col < 1024; ++col) {
    const float score = logits[col] + bias[col];
    if (score > first) {
      second = first;
      second_index = first_index;
      first = score;
      first_index = col;
    } else if (score > second) {
      second = score;
      second_index = col;
    }
  }
  top2[0] = first;
  top2[1] = static_cast<float>(first_index);
  top2[2] = second;
  top2[3] = static_cast<float>(second_index);
  for (int col = 4; col < 16; ++col)
    top2[col] = 0.0f;
}
