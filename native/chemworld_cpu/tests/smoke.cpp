#include "chemworld_cpu/c_api.h"

#include <cmath>
#include <cstdint>

int main() {
    cw_world *first = nullptr;
    cw_world *second = nullptr;
    if (cw_world_create(&first) != CW_OK || cw_world_create(&second) != CW_OK) {
        return 1;
    }
    cw_world *worlds[] = {first, second};
    const double inputs[] = {2.0, 3.0};
    double values[] = {0.0, 0.0};
    const int status = cw_world_step_batch(worlds, 2, 1, inputs, values);
    double observed = 0.0;
    std::uint64_t steps = 0;
    std::uint64_t seed = 0;
    const bool valid = status == CW_OK
        && cw_world_reset(first, 7) == CW_OK
        && cw_world_step(first, 1, 4.0, &observed) == CW_OK
        && cw_world_observe(first, &observed, &steps, &seed) == CW_OK
        && observed == 4.0 && steps == 1 && seed == 7
        && values[0] == 2.0 && values[1] == 3.0;
    cw_world_destroy(first);
    cw_world_destroy(second);
    return valid ? 0 : 1;
}
