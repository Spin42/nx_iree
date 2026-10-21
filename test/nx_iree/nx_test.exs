defmodule NxIREE.NxTest do
  use ExUnit.Case

  @illegal_ops [
    fft: 2,
    fft2: 2,
    ifft: 2,
    ifft2: 2,
    rfft: 2,
    irfft: 2,
    select_and_scatter: 3,
    window_scatter_min: 5,
    window_scatter_max: 5
  ]

  @rounding_error [
    erf: 1,
    erfc: 1,
    asinh: 1,
    cbrt: 1,
    atan2: 2,
    cosh: 1,
    expm1: 1,
    sigmoid: 1,
    standard_deviation: 2,
    atanh: 1,
    tanh: 1,
    tan: 1,
    real: 1,
    asin: 1,
    asinh: 1,
    atan: 1,
    acos: 1,
    acosh: 1,
    cos: 1,
    sin: 1,
    phase: 1,
    covariance: 3,
    weighted_mean: 3,
    variance: 2,
    # as_type tests on f64 which is not supported on metal
    as_type: 2,
    rsqrt: 1,
    sinh: 1,
    # f64 erf_inv is only accurate to about f32 precision in IREE
    erf_inv: 1
  ]

  @partial_support [
    # stable: true is not supported
    argsort: 2,
    # tensor/2 tests on f8 which is not supported yet
    tensor: 2,
    # bit_size/1 tests on u2 which is not supported yet
    bit_size: 1,
    # iota/2 tests on f64 which is not supported on metal
    iota: 2,
    # does not support complex tensors
    conv: 3
  ]

  @errors_to_be_fixed [
    # bug on filter_by_indices_list in the Nx compiler
    top_k: 2,
    # window_* crashes on iree-compile
    window_product: 3,
    window_mean: 3,
    window_sum: 3,
    window_min: 3,
    window_max: 3,
    # iree_linalg_ext.scatter rejects one of the doctests' indices layouts
    indexed_put: 4
  ]

  # Known gaps of IREE's SPIR-V code generation, measured with IREE 3.13 on
  # RADV (RDNA2 and RDNA3) and lavapipe
  @vulkan_known_issues [
    # f64 math does not compile for SPIR-V; NxIREE rejects f64 inputs there
    exp: 1,
    log: 1,
    log1p: 1,
    sqrt: 1,
    # f64 examples, and an f32 example that fails to compile
    all_close: 3,
    # f32 division within Vulkan's 2.5 ulp: 2.0 prints as 1.9999999 and
    # integer linspace truncates one step low
    divide: 2,
    mean: 2,
    log10: 1,
    logsumexp: 2,
    linspace: 3,
    # 8/16-bit integers are computed as 32-bit: clz and popcount count the
    # extra bits
    count_leading_zeros: 1,
    population_count: 1,
    # iree-compile fails
    mode: 2,
    remainder: 2,
    # wrong order on RDNA2 (RADV) only
    sort: 2
  ]

  @driver_exclusions if System.get_env("NX_IREE_DEFAULT_DRIVER") == "vulkan",
                       do: @vulkan_known_issues,
                       else: []

  doctest Nx,
    except:
      @illegal_ops ++
        @rounding_error ++ @partial_support ++ @errors_to_be_fixed ++ @driver_exclusions
end
