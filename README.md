# Loki accelerator template (LAT) neural network library

A simple neural network library which provides acceleration on the [Loki accelerator template](https://github.com/ucam-comparch-loki/lat-ifc).

Most compute functions come in two flavours:
1. Pre-allocated
 * Output data is written into a provided buffer
1. Allocated on demand
 * A buffer is allocated for the result and returned by the function
 * The user is responsible for calling `loki_free` on this buffer
 * The buffer is not guaranteed to have any particular dimension order

As with the LAT interface library, this library assumes that Loki has been configured with two cores and one accelerator on each tile.

## Prerequisites

Requires:
* [libloki](https://github.com/ucam-comparch-loki/libloki)
* [lat-ifc](https://github.com/ucam-comparch-loki/lat-ifc)

## Build

Requires the [Loki compiler](no_link_yet).

```
LIBLOKI_DIR=path/to/libloki LAT_IFC_DIR=path/to/lat-ifc make
```
