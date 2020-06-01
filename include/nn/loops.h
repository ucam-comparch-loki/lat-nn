#ifndef LAT_NN_LOOPS_H
#define LAT_NN_LOOPS_H

// Possible dimensions that loops can iterate over.
// Not all dimensions apply to all computations.
enum Loop {
  BATCH,
  IN_CHANNELS,
  OUT_CHANNELS,
  IMAGE_WIDTH,
  IMAGE_HEIGHT,
  FILTER_WIDTH_OS,  // output stationary
  FILTER_HEIGHT_OS, // output stationary
  FILTER_WIDTH_IS,  // input stationary
  FILTER_HEIGHT_IS  // input stationary
};

// A collection of loops, from outermost to innermost.
//
// Convolutions typically require
//   {BATCH, IN_CHANNELS, OUT_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, FILTER_WIDTH,
//    FILTER_HEIGHT}
// Linear layers typically require
//   {BATCH, IN_CHANNELS, OUT_CHANNELS}
//
// Loop nests may reasonably have more loops (e.g. if loop tiling is used) or
// fewer loops (e.g. if some dimensions are known to be size 1).
typedef struct {
  unsigned int loop_count;
  enum Loop* loops;
} loop_nest_t;


// Some sensible defaults.

// The way we'd probably write a convolution in code.
// Innermost loops are very short, so parallelism isn't great.
extern loop_nest_t LOOP_NEST_NAIVE;

// Output remains constant in both inner loops. Can accumulate in both
// accelerator dimensions.
extern loop_nest_t LOOP_NEST_OUTPUT_STATIONARY;

// Input remains constant in penultimate loop. Can broadcast along rows.
extern loop_nest_t LOOP_NEST_INPUT_STATIONARY;

// Weights remain constant in inner loops. Can broadcast along columns.
extern loop_nest_t LOOP_NEST_WEIGHT_STATIONARY;

#endif // include guard
