#define NOCPP

#include <aie_api/aie.hpp>

extern "C" void vocab_merge_top2_aie2(float *restrict candidates,
                                        float *restrict result) {
  float first = -1.0e30f;
  float second = -1.0e30f;
  int first_id = -1;
  int second_id = -1;
  for (int tile = 0; tile < 64; ++tile) {
    for (int rank = 0; rank < 2; ++rank) {
      const int offset = tile * 16 + rank * 2;
      const float score = candidates[offset];
      const int id = tile * 1024 + static_cast<int>(candidates[offset + 1]);
      if (score > first || (score == first && id < first_id)) {
        second = first;
        second_id = first_id;
        first = score;
        first_id = id;
      } else if (score > second || (score == second && id < second_id)) {
        second = score;
        second_id = id;
      }
    }
  }
  result[0] = first;
  result[1] = static_cast<float>(first_id);
  result[2] = second;
  result[3] = static_cast<float>(second_id);
}
