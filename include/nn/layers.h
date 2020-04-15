// TODO: Pull the neural-network specific stuff out of lat-ifc and into here.
// TODO: create a tensor type which generalises all of the `config_t`s and
//       exposes more information (e.g. size of each dimension).

#ifndef LAT_NN_LAYERS_H
#define LAT_NN_LAYERS_H

#include <lat/types.h>

typedef struct {
  uint32_t batchSize;
  uint32_t channels;
  uint32_t inputWidth;
  uint32_t inputHeight;
  uint32_t windowWidth;
  uint32_t windowHeight;
  uint32_t stride;        // In pixels.
} pool_shape_t;

// 2D convolution - the standard in CNNs for visual data.
void lat_conv2d(
  const activation_config_t* input,
  const filter_config_t* weights,
  activation_config_t* output,
  const conv_shape_t* params,
  uint32_t stride,    // TODO: put this in conv_shape_t?
  uint32_t dilation   // TODO: put this in conv_shape_t?
);

// Linear/fully-connected layer. Used for classification and multi-layer
// perceptrons.
// Currently requires 4D weights and activations, to match the convolution
// interface. Use 1 for any width/height parameters.
void lat_linear(
  const activation_config_t* input,
  const filter_config_t* weights,
  activation_config_t* output,
  uint32_t batchSize,  // TODO: should be able to infer this
  uint32_t numInputs,  // TODO: should be able to infer this
  uint32_t numOutputs  // TODO: should be able to infer this
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
  uint32_t stride,    // TODO: put this in conv_shape_t?
  uint32_t dilation   // TODO: put this in conv_shape_t?
);

// Linear/fully-connected layer with automatic allocation of output buffer.
// Currently requires 4D weights and activations, to match the convolution
// interface. Use 1 for any width/height parameters.
activation_config_t* lat_linear_alloc(
  const activation_config_t* input,
  const filter_config_t* weights,
  uint32_t batchSize,  // TODO: should be able to infer this
  uint32_t numInputs,  // TODO: should be able to infer this
  uint32_t numOutputs  // TODO: should be able to infer this
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
