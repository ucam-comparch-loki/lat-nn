// TODO: create a tensor type which generalises all of the `config_t`s and
//       exposes more information (e.g. size of each dimension).

#ifndef LAT_NN_TENSOR_H
#define LAT_NN_TENSOR_H

#include <lat/types.h>

// Details of 4D activations in memory.
typedef struct {
  memory_location_t data;

  // Distance (in bytes) between elements in each dimension. Negative offsets
  // are allowed (makes rotation/transpose trivial).
  int32_t  batch_stride;
  int32_t  channel_stride;
  int32_t  column_stride;
  int32_t  row_stride;
} activation_config_t;

// Details of 4D weights in memory.
typedef struct {
  memory_location_t data;

  // Distance (in bytes) between elements in each dimension. Negative offsets
  // are allowed (makes rotation/transpose trivial).
  int32_t  in_channel_stride;
  int32_t  out_channel_stride;
  int32_t  column_stride;
  int32_t  row_stride;

  // Not sure if this can be removed and replaced by a channel_stride and the
  // group size. It's certainly easier for it to remain separate.
  int32_t  group_stride;
} filter_config_t;

#endif // include guard
