#include <stdio.h>
#include <stdlib.h>
#include <lat/run.h>
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
data_t window_max(data_t* data, int num_cols, int col_step, int num_rows, int row_step) {
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

          // TODO: double check arguments.
          // I think the function expects `skip`s to be measured in elements,
          // but we're giving it bytes.
          *out_ptr = window_max(in_ptr, params->windowWidth, input->columnSkip,
                                params->windowHeight, input->rowSkip);
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
  printf("Error: avg_pool_2d not yet implemented\n");
  exit(1);
}


activation_config_t* lat_conv2d_alloc(
  const activation_config_t* input,
  const filter_config_t* weights,
  const conv_shape_t* params,
  uint32_t stride,
  uint32_t dilation
) {
  printf("Error: automatic allocation not yet implemented\n");
  exit(1);
  return NULL;
}

activation_config_t* lat_linear_alloc(
  const activation_config_t* input,
  const filter_config_t* weights,
  uint32_t batchSize,
  uint32_t numInputs,
  uint32_t numOutputs
) {
  printf("Error: automatic allocation not yet implemented\n");
  exit(1);
  return NULL;
}

activation_config_t* lat_max_pool_2d_alloc(
  const activation_config_t* input,
  const pool_shape_t* params
) {
  printf("Error: automatic allocation not yet implemented\n");
  exit(1);
  return NULL;
}

activation_config_t* lat_avg_pool_2d_alloc(
  const activation_config_t* input,
  const pool_shape_t* params
) {
  printf("Error: automatic allocation not yet implemented\n");
  exit(1);
  return NULL;
}
