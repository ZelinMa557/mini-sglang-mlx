#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include <assert.h>
#include "util.h"
#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif
namespace mx = mlx::core;

namespace mlx_serve {
mx::array save_kv_cache(
    const mx::array& kv,
    const mx::array& positions,
    const int dims,
    const float base,
    mx::StreamOrDevice s = {});
}