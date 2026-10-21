module attributes {transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint} {
  transform.named_sequence @apply_op_config(%op: !transform.any_op {transform.readonly}, %config: !transform.any_param {transform.readonly}) {
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_contraction(%op: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %c = transform.match.structured failures(propagate) %op : (!transform.any_op) -> !transform.any_op {
    ^bb0(%arg: !transform.any_op):
      transform.match.structured.classify_contraction_dims %arg : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
      // Plain M x N x K matmuls only; dot-product style reductions also
      // classify as contractions.
      %rank = transform.match.structured.rank %arg : (!transform.any_op) -> !transform.param<i64>
      %c3 = transform.param.constant 3 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c3 : !transform.param<i64>
      transform.match.structured.yield %arg : !transform.any_op
    }
    %config = transform.param.constant #iree_codegen.compilation_info<
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64, 8]]>,
      translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<MatmulPromoteVectorize> workgroup_size = [16, 8, 1], {pipeline_depth = 1 : i64, store_stage = 1 : i64}>
    > -> !transform.any_param
    transform.yield %c, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) -> !transform.any_op
      attributes {iree_codegen.tuning_spec_entrypoint} {
    %res = transform.foreach_match in %variant_op
        @match_contraction -> @apply_op_config
      : (!transform.any_op) -> !transform.any_op
    transform.yield %res : !transform.any_op
  }
}
