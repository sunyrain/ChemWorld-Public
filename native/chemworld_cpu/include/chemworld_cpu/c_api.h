#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef CHEMWORLD_CPU_BUILD
#    define CHEMWORLD_CPU_API __declspec(dllexport)
#  else
#    define CHEMWORLD_CPU_API __declspec(dllimport)
#  endif
#else
#  define CHEMWORLD_CPU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cw_world cw_world;

enum cw_status {
    CW_OK = 0,
    CW_INVALID_ARGUMENT = 1,
    CW_INVALID_OPERATION = 2,
    CW_NUMERIC_ERROR = 3,
    CW_INTERNAL_ERROR = 4,
};

enum cw_operation {
    CW_OPERATION_OBSERVE = 0,
    CW_OPERATION_ADD = 1,
    CW_OPERATION_SCALE = 2,
};

/* The ABI is intentionally C-shaped: no STL types or C++ object layout cross it. */
CHEMWORLD_CPU_API const char *cw_cpu_abi_version(void);
CHEMWORLD_CPU_API const char *cw_cpu_last_error(void);

CHEMWORLD_CPU_API int cw_world_create(cw_world **out_world);
CHEMWORLD_CPU_API void cw_world_destroy(cw_world *world);
CHEMWORLD_CPU_API int cw_world_reset(cw_world *world, uint64_t seed);
CHEMWORLD_CPU_API int cw_world_observe(
    const cw_world *world,
    double *value,
    uint64_t *steps,
    uint64_t *seed
);
CHEMWORLD_CPU_API int cw_world_step(
    cw_world *world,
    uint32_t operation,
    double input,
    double *value
);
CHEMWORLD_CPU_API int cw_world_step_batch(
    cw_world *const *worlds,
    size_t count,
    uint32_t operation,
    const double *inputs,
    double *values
);

#ifdef __cplusplus
}
#endif
