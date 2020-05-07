#include <stdio.h>
#include <stdlib.h>
#include <lat/run.h>
#include <loki/alloc.h>
#include <loki/channels.h>
#include <loki/ids.h>
#include "nn/layers.h"

void lat_conv2d(
  const activation_config_t* input,
  const filter_config_t* weights,
  activation_config_t* output,
  const conv_shape_t* params,
  uint32_t stride,
  uint32_t dilation
) {

  lat_parameters_t p;

  p.shape = *params;
  p.stride = stride;
  p.dilation = dilation;
  p.input = *input;
  p.output = *output;
  p.filters = *weights;
  uint32_t thisCore = single_core_bitmask(get_core_id());
  p.notificationAddress = loki_mcast_address(thisCore, CH_REGISTER_3, 0);

  lat_accelerate(&p);
  // Could do something else while waiting. (But need to put `p` on the heap.)
  // Or could notify a different core, in a software pipeline sort of way.
  lat_sync(&p);

}

void lat_linear(
  const activation_config_t* input,
  const filter_config_t* weights,
  activation_config_t* output,
  uint32_t batchSize,
  uint32_t numInputs,
  uint32_t numOutputs
) {
  // TODO: the accelerator interface is currently limited to convolutions.
  // Simplify this when the interface is generalised.

  // Encode this layer as a convolution.
  conv_shape_t conv;
  conv.batchSize = batchSize;
  conv.inChannels = numInputs;
  conv.outChannels = numOutputs;
  conv.imageWidth = 1;
  conv.imageHeight = 1;
  conv.filterWidth = 1;
  conv.filterHeight = 1;
  conv.groups = 1;

  lat_conv2d(input, weights, output, &conv, 1, 0);
}


// This is for one window.
data_t window_max(data_t* data, int num_cols, int col_step, int num_rows,
                  int row_step) {
  data_t max = *data;

  for (int row=0; row<num_rows; row++) {
    data_t* data_ptr = data + (row * row_step);

    for (int col=0; col<num_cols; col++) {
      if (*data_ptr > max)
        max = *data_ptr;

      data_ptr += col_step;
    }
  }

  return max;
}

// This is for one window.
data_t window_avg(data_t* data, int num_cols, int col_step, int num_rows,
                  int row_step) {
  data_t sum = 0;

  for (int row=0; row<num_rows; row++) {
    data_t* data_ptr = data + (row * row_step);

    for (int col=0; col<num_cols; col++) {
      sum += *data_ptr;
      data_ptr += col_step;
    }
  }

  // Note that Loki doesn't have a division unit, so this is especially slow.
  return sum / (num_cols * num_rows);
}

void lat_max_pool_2d(
  const activation_config_t* input,
  activation_config_t* output,
  const pool_shape_t* params
) {
  // TODO: Have a way to turn warnings off.
  fprintf(stderr, "Note: max pool is running on Loki CPU (unoptimised)\n");

  // This is the furthest we can iterate through the input, while still having
  // space for a full window.
  uint maxHeight = params->inputHeight - params->windowHeight + 1;
  uint maxWidth = params->inputWidth - params->windowWidth + 1;

  for (uint b=0; b<params->batchSize; b++) {
    for (uint ch=0; ch<params->channels; ch++) {
      for (uint row=0; row<maxHeight; row+=params->stride) {
        for (uint col=0; col<maxWidth; col+=params->stride) {
          data_t* in_ptr = input->address + b*input->batchSkip +
                           ch*input->channelSkip + row*input->rowSkip +
                           col*input->columnSkip;
          data_t* out_ptr = output->address + b*output->batchSkip +
                            ch*output->channelSkip + row*output->rowSkip +
                            col*output->columnSkip;

          *out_ptr = window_max(in_ptr, params->windowWidth,
                                input->columnSkip / sizeof(data_t),
                                params->windowHeight,
                                input->rowSkip / sizeof(data_t));
        }
      }
    }
  }
}

void lat_avg_pool_2d(
  const activation_config_t* input,
  activation_config_t* output,
  const pool_shape_t* params
) {
  // TODO: Have a way to turn warnings off.
  fprintf(stderr, "Note: avg pool is running on Loki CPU (unoptimised)\n");

  // This is the furthest we can iterate through the input, while still having
  // space for a full window.
  uint maxHeight = params->inputHeight - params->windowHeight + 1;
  uint maxWidth = params->inputWidth - params->windowWidth + 1;

  for (uint b=0; b<params->batchSize; b++) {
    for (uint ch=0; ch<params->channels; ch++) {
      for (uint row=0; row<maxHeight; row+=params->stride) {
        for (uint col=0; col<maxWidth; col+=params->stride) {
          data_t* in_ptr = input->address + b*input->batchSkip +
                           ch*input->channelSkip + row*input->rowSkip +
                           col*input->columnSkip;
          data_t* out_ptr = output->address + b*output->batchSkip +
                            ch*output->channelSkip + row*output->rowSkip +
                            col*output->columnSkip;

          *out_ptr = window_avg(in_ptr, params->windowWidth,
                                input->columnSkip / sizeof(data_t),
                                params->windowHeight,
                                input->rowSkip / sizeof(data_t));
        }
      }
    }
  }
}


// Determine how large the output will be (in pixels), given the computation
// parameters. This applies to any windowed computation, e.g. convolution,
// pooling.
// The equation is taken from PyTorch (minus the `padding` parameter):
// https://pytorch.org/docs/stable/nn.html#conv2d
uint output_size(uint input_size, uint window_size, uint stride, uint dilation) {
  return ((input_size - dilation * (window_size - 1) - 1) / stride) + 1;
}

// Allocate and initialise an activation tensor with the given dimensions.
// The default dimension order is [batch, channels, height, width].
// The user is responsible for deallocating both the tensor and its data array
// (`tensor->address`).
activation_config_t* init_activation_tensor(uint batch, uint channels,
                                            uint height, uint width) {
  activation_config_t* tensor = loki_malloc(sizeof(activation_config_t));
  assert(tensor != NULL);

  tensor->address = loki_malloc(batch * channels * width * height *
                                sizeof(data_t));
  assert(tensor->address != NULL);

  tensor->rowSkip = sizeof(data_t);
  tensor->columnSkip = width * tensor->rowSkip;
  tensor->channelSkip = height * tensor->columnSkip;
  tensor->batchSkip = channels * tensor->channelSkip;

  return tensor;
}


activation_config_t* lat_conv2d_alloc(
  const activation_config_t* input,
  const filter_config_t* weights,
  const conv_shape_t* params,
  uint32_t stride,
  uint32_t dilation
) {
  uint batch = params->batchSize;
  uint channels = params->outChannels;
  uint height = output_size(params->imageHeight, params->filterHeight, stride, dilation);
  uint width = output_size(params->imageWidth, params->filterWidth, stride, dilation);

  activation_config_t* output =
      init_activation_tensor(batch, channels, height, width);

  // Default: use same memory group as `input`.
  output->memoryConfigEncoded = input->memoryConfigEncoded;

  lat_conv2d(input, weights, output, params, stride, dilation);

  return output;
}

activation_config_t* lat_linear_alloc(
  const activation_config_t* input,
  const filter_config_t* weights,
  uint32_t batchSize,
  uint32_t numInputs,
  uint32_t numOutputs
) {
  // Add dummy dimensions so this tensor can be passed to the convolution
  // function.
  activation_config_t* output =
      init_activation_tensor(batchSize, numOutputs, 1, 1);

  // Default: use same memory group as `input`.
  output->memoryConfigEncoded = input->memoryConfigEncoded;

  lat_linear(input, weights, output, batchSize, numInputs, numOutputs);

  return output;
}

activation_config_t* lat_max_pool_2d_alloc(
  const activation_config_t* input,
  const pool_shape_t* params
) {
  uint batch = params->batchSize;
  uint channels = params->channels;
  uint height = output_size(params->inputHeight, params->windowHeight, params->stride, 1);
  uint width = output_size(params->inputWidth, params->windowWidth, params->stride, 1);

  activation_config_t* output =
      init_activation_tensor(batch, channels, height, width);

  // Default: use same memory group as `input`.
  output->memoryConfigEncoded = input->memoryConfigEncoded;

  lat_max_pool_2d(input, output, params);

  return output;
}

activation_config_t* lat_avg_pool_2d_alloc(
  const activation_config_t* input,
  const pool_shape_t* params
) {
  uint batch = params->batchSize;
  uint channels = params->channels;
  uint height = output_size(params->inputHeight, params->windowHeight, params->stride, 1);
  uint width = output_size(params->inputWidth, params->windowWidth, params->stride, 1);

  activation_config_t* output =
      init_activation_tensor(batch, channels, height, width);

  // Default: use same memory group as `input`.
  output->memoryConfigEncoded = input->memoryConfigEncoded;

  lat_avg_pool_2d(input, output, params);

  return output;
}
