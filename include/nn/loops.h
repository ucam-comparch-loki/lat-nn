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
  FILTER_WIDTH,
  FILTER_HEIGHT
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
extern loop_nest_t LOOP_NEST_NAIVE;
extern loop_nest_t LOOP_NEST_OUTPUT_STATIONARY;
extern loop_nest_t LOOP_NEST_INPUT_STATIONARY;
extern loop_nest_t LOOP_NEST_WEIGHT_STATIONARY;

#endif // include guard
