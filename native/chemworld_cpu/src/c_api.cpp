#include "chemworld_cpu/c_api.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace {

thread_local std::string last_error;

struct World {
    double value = 0.0;
    std::uint64_t steps = 0;
    std::uint64_t seed = 0;
};

bool finite(double value) {
    return std::isfinite(value) != 0;
}

int fail(int status, const char *message) {
    last_error = message;
    return status;
}

int apply(World &world, std::uint32_t operation, double input) {
    if (!finite(input)) {
        return fail(CW_NUMERIC_ERROR, "world input must be finite");
    }
    double next = world.value;
    switch (operation) {
    case 0: // observe-only step; useful for binding and scheduling smoke tests.
        break;
    case 1: // additive operation in the canonical dimensionless fixture state.
        next += input;
        break;
    case 2: // affine scale operation; avoids an implicit physics claim.
        next *= input;
        break;
    default:
        return fail(CW_INVALID_OPERATION, "unknown CPU fixture operation");
    }
    if (!finite(next)) {
        return fail(CW_NUMERIC_ERROR, "world result is not finite");
    }
    world.value = next;
    ++world.steps;
    return CW_OK;
}

} // namespace

struct cw_world {
    World state;
};

extern "C" {

CHEMWORLD_CPU_API const char *cw_cpu_abi_version(void) {
    return "chemworld-cpu-abi-1";
}

CHEMWORLD_CPU_API const char *cw_cpu_last_error(void) {
    return last_error.c_str();
}

CHEMWORLD_CPU_API int cw_world_create(cw_world **out_world) {
    if (out_world == nullptr) {
        return fail(CW_INVALID_ARGUMENT, "out_world is required");
    }
    try {
        *out_world = new cw_world{};
    } catch (...) {
        *out_world = nullptr;
        return fail(CW_INTERNAL_ERROR, "world allocation failed");
    }
    last_error.clear();
    return CW_OK;
}

CHEMWORLD_CPU_API void cw_world_destroy(cw_world *world) {
    delete world;
}

CHEMWORLD_CPU_API int cw_world_reset(cw_world *world, std::uint64_t seed) {
    if (world == nullptr) {
        return fail(CW_INVALID_ARGUMENT, "world is required");
    }
    world->state = World{};
    world->state.seed = seed;
    last_error.clear();
    return CW_OK;
}

CHEMWORLD_CPU_API int cw_world_observe(
    const cw_world *world,
    double *value,
    std::uint64_t *steps,
    std::uint64_t *seed
) {
    if (world == nullptr || value == nullptr || steps == nullptr || seed == nullptr) {
        return fail(CW_INVALID_ARGUMENT, "world and all observe outputs are required");
    }
    *value = world->state.value;
    *steps = world->state.steps;
    *seed = world->state.seed;
    last_error.clear();
    return CW_OK;
}

CHEMWORLD_CPU_API int cw_world_step(
    cw_world *world,
    std::uint32_t operation,
    double input,
    double *value
) {
    if (world == nullptr || value == nullptr) {
        return fail(CW_INVALID_ARGUMENT, "world and output value are required");
    }
    const int status = apply(world->state, operation, input);
    if (status == CW_OK) {
        *value = world->state.value;
        last_error.clear();
    }
    return status;
}

CHEMWORLD_CPU_API int cw_world_step_batch(
    cw_world *const *worlds,
    std::size_t count,
    std::uint32_t operation,
    const double *inputs,
    double *values
) {
    if (count == 0) {
        last_error.clear();
        return CW_OK;
    }
    if (worlds == nullptr || inputs == nullptr || values == nullptr) {
        return fail(CW_INVALID_ARGUMENT, "batch worlds, inputs and outputs are required");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (worlds[index] == nullptr) {
            return fail(CW_INVALID_ARGUMENT, "batch contains a null world");
        }
        const int status = apply(worlds[index]->state, operation, inputs[index]);
        if (status != CW_OK) {
            return status;
        }
        values[index] = worlds[index]->state.value;
    }
    last_error.clear();
    return CW_OK;
}

} // extern "C"
