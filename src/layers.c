#include <stdio.h>
#include <stdlib.h>
#include <lat/run.h>
#include <loki/alloc.h>
#include <loki/channels.h>
#include <loki/channel_io.h>
#include <loki/ids.h>
#include "nn/layers.h"

// Determine how large the output will be (in pixels), given the computation
// parameters. This applies to any windowed computation, e.g. convolution,
// pooling.
// The equation is taken from PyTorch (minus the `padding` parameter):
// https://pytorch.org/docs/stable/nn.html#conv2d
uint output_size(uint input_size, uint window_size, uint stride, uint dilation) {
  int size = ((input_size - dilation * (window_size - 1) - 1) / stride) + 1;
  return (size < 0) ? 0 : size;
}

void lat_conv2d(
  const activation_config_t* input,
  const filter_config_t* weights,
  activation_config_t* output,
  const conv_shape_t* params,
  const loop_nest_t* loop_order
) {

  // TODO: provide a default loop order if none is provided.
  assert(loop_order != NULL);

  // Memory allocation is not multi-tile safe, so use statically allocated
  // arrays.
  assert(loop_order->loop_count <= 10);
  loop_iteration_t loops[10];
  uint32_t iteration_counts[10];

  lat_parameters_t p;

  uint32_t this_core = single_core_bitmask(get_core_id());
  p.notification_address = loki_mcast_address(this_core, CH_REGISTER_3, 0);

  p.loop_count = loop_order->loop_count;
  p.loops = loops;
  p.iteration_counts = iteration_counts;

  // Default: in1=input, in2=weights, out=output.
  for (uint i=0; i<p.loop_count; i++) {
    switch (loop_order->loops[i]) {
      case BATCH:
        p.loops[i].in1_stride = input->batch_stride;
        p.loops[i].in2_stride = 0;
        p.loops[i].out_stride = output->batch_stride;
        p.iteration_counts[i] = params->batch_size;
        break;

      case IN_CHANNELS:
        p.loops[i].in1_stride = input->channel_stride;
        p.loops[i].in2_stride = weights->in_channel_stride;
        p.loops[i].out_stride = 0;
        p.iteration_counts[i] = params->in_channels;
        break;

      case OUT_CHANNELS:
        p.loops[i].in1_stride = 0;
        p.loops[i].in2_stride = weights->out_channel_stride;
        p.loops[i].out_stride = output->channel_stride;
        p.iteration_counts[i] = params->out_channels;
        break;

      case IMAGE_WIDTH:
        p.loops[i].in1_stride = input->column_stride * params->stride;
        p.loops[i].in2_stride = 0;
        p.loops[i].out_stride = output->column_stride;
        p.iteration_counts[i] = output_size(params->image_width,
            params->filter_width, params->stride, params->dilation);
        break;

      case IMAGE_HEIGHT:
        p.loops[i].in1_stride = input->row_stride * params->stride;
        p.loops[i].in2_stride = 0;
        p.loops[i].out_stride = output->row_stride;
        p.iteration_counts[i] = output_size(params->image_height,
            params->filter_height, params->stride, params->dilation);
        break;

      case FILTER_WIDTH:
        p.loops[i].in1_stride = input->column_stride * params->dilation;
        p.loops[i].in2_stride = weights->column_stride;
        p.loops[i].out_stride = 0;
        p.iteration_counts[i] = params->filter_width;
        break;

      case FILTER_HEIGHT:
        p.loops[i].in1_stride = input->row_stride * params->dilation;
        p.loops[i].in2_stride = weights->row_stride;
        p.loops[i].out_stride = 0;
        p.iteration_counts[i] = params->filter_height;
        break;

      default:
        printf("Error: unsupported convolution Loop enum: %d\n",
               loop_order->loops[i]);
        exit(1);
        break;
    }
  }

  p.in1 = input->data;
  p.in2 = weights->data;
  p.out = output->data;

  lat_accelerate(&p);
  // Could do something else while waiting. (But need to put `p` on the heap.)
  // Or could notify a different core, in a software pipeline sort of way.
  lat_sync(&p);

}

void lat_linear(
  const activation_config_t* input,
  const filter_config_t* weights,
  activation_config_t* output,
  uint32_t batch_size,
  uint32_t num_inputs,
  uint32_t num_outputs,
  const loop_nest_t* loop_order
) {
  // TODO: the accelerator interface is currently limited to convolutions.
  // Simplify this when the interface is generalised.

  // Encode this layer as a convolution.
  conv_shape_t conv;
  conv.batch_size = batch_size;
  conv.in_channels = num_inputs;
  conv.out_channels = num_outputs;
  conv.image_width = 1;
  conv.image_height = 1;
  conv.filter_width = 1;
  conv.filter_height = 1;
  conv.groups = 1;
  conv.stride = 1;
  conv.dilation = 1;

  lat_conv2d(input, weights, output, &conv, loop_order);
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
  // TODO: dilation
  uint max_height = params->input_height - params->window_height + 1;
  uint max_width = params->input_width - params->window_width + 1;

  for (uint b=0; b<params->batch_size; b++) {
    for (uint ch=0; ch<params->channels; ch++) {
      for (uint row=0; row<max_height; row+=params->stride) {
        for (uint col=0; col<max_width; col+=params->stride) {
          data_t* in_ptr = input->data.address + (b*input->batch_stride +
                           ch*input->channel_stride + row*input->row_stride +
                           col*input->column_stride) / sizeof(data_t);
          data_t* out_ptr = output->data.address + (b*output->batch_stride +
                            ch*output->channel_stride + row*output->row_stride +
                            col*output->column_stride) / sizeof(data_t);

          *out_ptr = window_max(in_ptr, params->window_width,
                                input->column_stride / sizeof(data_t),
                                params->window_height,
                                input->row_stride / sizeof(data_t));
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
  // TODO: dilation.
  uint max_height = params->input_height - params->window_height + 1;
  uint max_width = params->input_width - params->window_width + 1;

  for (uint b=0; b<params->batch_size; b++) {
    for (uint ch=0; ch<params->channels; ch++) {
      for (uint row=0; row<max_height; row+=params->stride) {
        for (uint col=0; col<max_width; col+=params->stride) {
          data_t* in_ptr = input->data.address + (b*input->batch_stride +
                           ch*input->channel_stride + row*input->row_stride +
                           col*input->column_stride) / sizeof(data_t);
          data_t* out_ptr = output->data.address + (b*output->batch_stride +
                            ch*output->channel_stride + row*output->row_stride +
                            col*output->column_stride) / sizeof(data_t);

          *out_ptr = window_avg(in_ptr, params->window_width,
                                input->column_stride / sizeof(data_t),
                                params->window_height,
                                input->row_stride / sizeof(data_t));
        }
      }
    }
  }
}


// Allocate and initialise an activation tensor with the given dimensions.
// The default dimension order is [batch, channels, height, width].
// The user is responsible for deallocating both the tensor and its data array
// (`tensor->address`).
activation_config_t* init_activation_tensor(uint batch, uint channels,
                                            uint height, uint width) {
  activation_config_t* tensor = loki_malloc(sizeof(activation_config_t));
  assert(tensor != NULL);

  size_t words = batch * channels * width * height;
  size_t bytes = words * sizeof(data_t);
  tensor->data.address = loki_malloc(bytes);
  assert(tensor->data.address != NULL);

  tensor->row_stride = sizeof(data_t);
  tensor->column_stride = width * tensor->row_stride;
  tensor->channel_stride = height * tensor->column_stride;
  tensor->batch_stride = channels * tensor->channel_stride;

  return tensor;
}

// Set values in memory to 0. Useful for anything which accumulates results in
// memory, e.g. convolutions. Not necessary for functions which write the result
// directly, e.g. pooling.
// Warning: overwrites output channel 2 (as allowed by the ABI).
void clear_memory(data_t* address, size_t num_words, int memory_config) {
  set_channel_map(2, memory_config);
  loki_channel_memset_words(2, address, 0, num_words);
}

activation_config_t* lat_conv2d_alloc(
  const activation_config_t* input,
  const filter_config_t* weights,
  const conv_shape_t* params,
  const loop_nest_t* loop_order
) {
  uint batch = params->batch_size;
  uint channels = params->out_channels;
  uint height = output_size(params->image_height, params->filter_height, params->stride, params->dilation);
  uint width = output_size(params->image_width, params->filter_width, params->stride, params->dilation);

  activation_config_t* output =
      init_activation_tensor(batch, channels, height, width);

  // Default: use same memory group as `input`.
  output->data.memory_config = input->data.memory_config;

  // Initialise tensor contents to zero.
  clear_memory(output->data.address, batch*channels*height*width*sizeof(data_t)/4, output->data.memory_config);

  lat_conv2d(input, weights, output, params, loop_order);

  return output;
}

activation_config_t* lat_linear_alloc(
  const activation_config_t* input,
  const filter_config_t* weights,
  uint32_t batch_size,
  uint32_t num_inputs,
  uint32_t num_outputs,
  const loop_nest_t* loop_order
) {
  // Add dummy dimensions so this tensor can be passed to the convolution
  // function.
  activation_config_t* output =
      init_activation_tensor(batch_size, num_outputs, 1, 1);

  // Default: use same memory group as `input`.
  output->data.memory_config = input->data.memory_config;

  // Initialise tensor contents to zero.
  clear_memory(output->data.address, num_outputs*sizeof(data_t)/4, output->data.memory_config);

  lat_linear(input, weights, output, batch_size, num_inputs, num_outputs, loop_order);

  return output;
}

activation_config_t* lat_max_pool_2d_alloc(
  const activation_config_t* input,
  const pool_shape_t* params
) {
  uint batch = params->batch_size;
  uint channels = params->channels;
  uint height = output_size(params->input_height, params->window_height, params->stride, 1);
  uint width = output_size(params->input_width, params->window_width, params->stride, 1);

  activation_config_t* output =
      init_activation_tensor(batch, channels, height, width);

  // Default: use same memory group as `input`.
  output->data.memory_config = input->data.memory_config;

  lat_max_pool_2d(input, output, params);

  return output;
}

activation_config_t* lat_avg_pool_2d_alloc(
  const activation_config_t* input,
  const pool_shape_t* params
) {
  uint batch = params->batch_size;
  uint channels = params->channels;
  uint height = output_size(params->input_height, params->window_height, params->stride, 1);
  uint width = output_size(params->input_width, params->window_width, params->stride, 1);

  activation_config_t* output =
      init_activation_tensor(batch, channels, height, width);

  // Default: use same memory group as `input`.
  output->data.memory_config = input->data.memory_config;

  lat_avg_pool_2d(input, output, params);

  return output;
}
