#ifndef LAT_NN_LAYERS_H
#define LAT_NN_LAYERS_H

#include <lat/types.h>
#include "loops.h"
#include "tensor.h"

// Size and shape of convolution.
typedef struct {
  uint32_t batch_size;
  uint32_t in_channels;
  uint32_t out_channels;
  uint32_t image_width;
  uint32_t image_height;
  uint32_t filter_width;
  uint32_t filter_height;

  // Channels (both in and out) partitioned into this many groups: default 1.
  uint32_t groups;

  // Step size (in pixels) between adjacent filter positions: default 1.
  uint32_t stride;

  // Distance between activation pixels multiplied by weights: default 1.
  uint32_t dilation;
} conv_shape_t;

typedef struct {
  uint32_t batch_size;
  uint32_t channels;
  uint32_t input_width;
  uint32_t input_height;
  uint32_t window_width;
  uint32_t window_height;
  uint32_t stride;        // In pixels.
} pool_shape_t;

// 2D convolution - the standard in CNNs for visual data.
void lat_conv2d(
  const activation_config_t* input,
  const filter_config_t* weights,
  activation_config_t* output,
  const conv_shape_t* params,
  const loop_nest_t* loop_order
);

// Linear/fully-connected layer. Used for classification and multi-layer
// perceptrons.
// Currently requires 4D weights and activations, to match the convolution
// interface. Use 1 for any width/height parameters.
void lat_linear(
  const activation_config_t* input,
  const filter_config_t* weights,
  activation_config_t* output,
  uint32_t batch_size,
  uint32_t num_inputs,
  uint32_t num_outputs,
  const loop_nest_t* loop_order
);

// Downsample input by taking the maximum value in each window.
void lat_max_pool_2d(
  const activation_config_t* input,
  activation_config_t* output,
  const pool_shape_t* params
);

// Downsample input by taking the average value in each window.
void lat_avg_pool_2d(
  const activation_config_t* input,
  activation_config_t* output,
  const pool_shape_t* params
);


// 2D convolution with automatic allocation of output buffer.
activation_config_t* lat_conv2d_alloc(
  const activation_config_t* input,
  const filter_config_t* weights,
  const conv_shape_t* params,
  const loop_nest_t* loop_order
);

// Linear/fully-connected layer with automatic allocation of output buffer.
// Currently requires 4D weights and activations, to match the convolution
// interface. Use 1 for any width/height parameters.
activation_config_t* lat_linear_alloc(
  const activation_config_t* input,
  const filter_config_t* weights,
  uint32_t batch_size,
  uint32_t num_inputs,
  uint32_t num_outputs,
  const loop_nest_t* loop_order
);

// Downsample input by taking the maximum value in each window.
activation_config_t* lat_max_pool_2d_alloc(
  const activation_config_t* input,
  const pool_shape_t* params
);

// Downsample input by taking the average value in each window.
activation_config_t* lat_avg_pool_2d_alloc(
  const activation_config_t* input,
  const pool_shape_t* params
);

#endif // include guard
