#include <stdio.h>

#include <inttypes.h>

#ifndef tile_size_x_qd
  #define tile_size_x_qd %(tile_size_x_qd)s
#endif

#ifndef block_size_x_qd
  #define block_size_x_qd %(block_size_x_qd)s
#endif

#ifndef block_size_y_qd
  #define block_size_y_qd %(block_size_y_qd)s
#endif

#ifndef window_width
#define window_width %(window_width)s
#endif

#define USE_READ_ONLY_CACHE read_only
#if USE_READ_ONLY_CACHE == 1
#define LDG(x, y) __ldg(x+y)
#elif USE_READ_ONLY_CACHE == 0
#define LDG(x, y) x[y]
#endif

#ifndef write_sums_qd
#define write_sums_qd %(write_sums_qd)s
#endif

#ifndef block_size_x_d
#define block_size_x_d %(block_size_x_d)s
#endif
