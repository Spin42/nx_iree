"""Write an IREE tuning spec that sets the SPIR-V configuration of selected
op families, leaving every other dispatch to the compiler's defaults.

Usage: family_spec.py out.mlir RULE...

Matmul rule:   DIMS:TM,TN,TK:WX,WY
    Rank-3 contractions whose loop dimensions match DIMS get workgroup tiles
    TM x TN x TK on a WX x WY workgroup with MatmulPromoteVectorize; each
    thread computes (TM / WY) x (TN / WX) outputs.
Generic rule:  g:DIMS:WGTILES/THREADTILES:WX,WY,WZ
    Ops of rank len(WGTILES) matching DIMS get BaseVectorize with those
    workgroup and thread tile lists.

DIMS is a comma list of index=value checks on the op's loop dimensions, for
example 0=400 (M is 400) or 0=6400,1=128. The first matching rule wins.
"""
import sys


def checks(dims):
    out = []
    for j, check in enumerate(dims.split(",")):
        index, value = check.split("=")
        out.append(f"""      %d{j} = transform.match.structured.dim %arg[{index}] : (!transform.any_op) -> !transform.param<i64>
      %v{j} = transform.param.constant {value} : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %d{j}, %v{j} : !transform.param<i64>""")
    return "\n".join(out)


def rule_body(rule):
    if rule.startswith("g:"):
        _, dims, tiles, wg = rule.split(":")
        wg_tiles, thread_tiles = tiles.split("/")
        rank = len(wg_tiles.split(","))
        classify = ""
        config = f"""lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{wg_tiles}], [{thread_tiles}]]>,
      translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseVectorize> workgroup_size = [{wg}]>"""
    else:
        dims, tiles, wg = rule.split(":")
        rank = 3
        classify = "      transform.match.structured.classify_contraction_dims %arg : (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)\n"
        config = f"""lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tiles}]]>,
      translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<MatmulPromoteVectorize> workgroup_size = [{wg}, 1], {{pipeline_depth = 1 : i64, store_stage = 1 : i64}}>"""
    return f"""    %c = transform.match.structured failures(propagate) %op : (!transform.any_op) -> !transform.any_op {{
    ^bb0(%arg: !transform.any_op):
{classify}      %rank = transform.match.structured.rank %arg : (!transform.any_op) -> !transform.param<i64>
      %r = transform.param.constant {rank} : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %r : !transform.param<i64>
{checks(dims)}
      transform.match.structured.yield %arg : !transform.any_op
    }}
    %config = transform.param.constant #iree_codegen.compilation_info<
      {config}
    > -> !transform.any_param
    transform.yield %c, %config : !transform.any_op, !transform.any_param"""


out, rules = sys.argv[1], sys.argv[2:]
if not rules:
    sys.exit(__doc__)
sequences = "".join(
    f"""
  transform.named_sequence @match_{i}(%op: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
{rule_body(rule)}
  }}
""" for i, rule in enumerate(rules))
calls = ", ".join(f"@match_{i} -> @apply_op_config" for i in range(len(rules)))
with open(out, "w") as f:
    f.write(f"""module attributes {{transform.with_named_sequence, iree_codegen.tuning_spec_with_default_entrypoint}} {{
  transform.named_sequence @apply_op_config(%op: !transform.any_op {{transform.readonly}}, %config: !transform.any_param {{transform.readonly}}) {{
    transform.annotate %op "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.yield
  }}
{sequences}
  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {{transform.consumed}}) -> !transform.any_op
      attributes {{iree_codegen.tuning_spec_entrypoint}} {{
    %res = transform.foreach_match in %variant_op {calls} : (!transform.any_op) -> !transform.any_op
    transform.yield %res : !transform.any_op
  }}
}}
""")
