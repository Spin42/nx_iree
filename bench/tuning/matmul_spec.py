"""Write an IREE tuning spec that sets the SPIR-V matmul configuration for
every contraction.

Usage: matmul_spec.py <out.mlir> <tm> <tn> <tk> <wg_x> <wg_y> <pipeline_depth> [store_stage]

Tiles are workgroup tile sizes (M, N, K); each thread computes
(tm / wg_y) x (tn / wg_x) outputs.
"""
import sys

out, tm, tn, tk, wx, wy, depth = sys.argv[1:8]
store_stage = sys.argv[8] if len(sys.argv) > 8 else "1"

spec = f"""module attributes {{transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint}} {{
  transform.named_sequence @apply_op_config(%op: !transform.any_op {{transform.readonly}}, %config: !transform.any_param {{transform.readonly}}) {{
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.yield
  }}

  transform.named_sequence @match_contraction(%op: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
    %c = transform.match.structured failures(propagate) %op : (!transform.any_op) -> !transform.any_op {{
    ^bb0(%arg: !transform.any_op):
      transform.match.structured.classify_contraction_dims %arg : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
      // Plain M x N x K matmuls only; dot-product style reductions also
      // classify as contractions.
      %rank = transform.match.structured.rank %arg : (!transform.any_op) -> !transform.param<i64>
      %c3 = transform.param.constant 3 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c3 : !transform.param<i64>
      transform.match.structured.yield %arg : !transform.any_op
    }}
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tm}, {tn}, {tk}]]>,
      translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<MatmulPromoteVectorize> workgroup_size = [{wx}, {wy}, 1], {{pipeline_depth = {depth} : i64, store_stage = {store_stage} : i64}}>
    > -> !transform.any_param
    transform.yield %c, %config : !transform.any_op, !transform.any_param
  }}

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {{transform.consumed}}) -> !transform.any_op
      attributes {{iree_codegen.tuning_spec_entrypoint}} {{
    %res = transform.foreach_match in %variant_op
        @match_contraction -> @apply_op_config
      : (!transform.any_op) -> !transform.any_op
    transform.yield %res : !transform.any_op
  }}
}}
"""
open(out, "w").write(spec)
