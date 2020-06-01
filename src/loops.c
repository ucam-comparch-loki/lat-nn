#include "nn/loops.h"

// The way we'd probably write a convolution in code.
// Innermost loops are very short, so parallelism isn't great.
enum Loop LOOPS_NAIVE[7] = {BATCH, OUT_CHANNELS, IN_CHANNELS, IMAGE_HEIGHT,
    IMAGE_WIDTH, FILTER_HEIGHT_OS, FILTER_WIDTH_OS};

// Output remains constant in both inner loops. Can accumulate in both
// accelerator dimensions.
enum Loop LOOPS_OUTPUT_STATIONARY[7] = {BATCH, OUT_CHANNELS, IMAGE_HEIGHT,
    IMAGE_WIDTH, FILTER_HEIGHT_OS, FILTER_WIDTH_OS, IN_CHANNELS};

// Input remains constant in penultimate loop. Can broadcast along rows.
// (Columns too, but input isn't constant there.)
enum Loop LOOPS_INPUT_STATIONARY[7] = {BATCH, IN_CHANNELS, IMAGE_HEIGHT,
    IMAGE_WIDTH, FILTER_HEIGHT_IS, OUT_CHANNELS, FILTER_WIDTH_IS};

// Weights remain constant in inner loops. Can broadcast in both accelerator
// dimensions.
enum Loop LOOPS_WEIGHT_STATIONARY[7] = {OUT_CHANNELS, IN_CHANNELS,
    FILTER_HEIGHT_OS, FILTER_WIDTH_OS, BATCH, IMAGE_HEIGHT, IMAGE_WIDTH};


loop_nest_t LOOP_NEST_NAIVE = {
  .loop_count = 7,
  .loops = LOOPS_NAIVE
};

loop_nest_t LOOP_NEST_OUTPUT_STATIONARY = {
  .loop_count = 7,
  .loops = LOOPS_OUTPUT_STATIONARY
};

loop_nest_t LOOP_NEST_INPUT_STATIONARY = {
  .loop_count = 7,
  .loops = LOOPS_INPUT_STATIONARY
};

loop_nest_t LOOP_NEST_WEIGHT_STATIONARY = {
  .loop_count = 7,
  .loops = LOOPS_WEIGHT_STATIONARY
};
