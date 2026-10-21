# Pretrained models on nx_iree

`bench/real_models.exs` runs pretrained HuggingFace checkpoints through
Bumblebee servings on EXLA (the reference) and on nx_iree, and compares what
an application sees. Host: Ryzen CPU and Radeon RX (RDNA, RADV), IREE 3.13,
generic Vulkan target, parameters preallocated on the device, preprocessing
compiled through the global defn options. Times are per call after
compilation, through a started serving.

| Check | Comparison | EXLA CPU | nx_iree CPU | nx_iree Vulkan |
| --- | --- | --- | --- | --- |
| DistilBERT SST-2, 4 padded sentences | labels and scores (diff 0) | 34 ms | 21 ms | 13 ms |
| BERT-base fill-mask | top-5 tokens (score diff ≤ 1.5e-5) | | | |
| GPT-2, 24 greedy tokens | identical text | 309 ms | 306 ms | 560 ms |
| ResNet-50, 640×480 photo | top-5 labels (diff 0) | 35 ms | 71 ms | 134 ms |
| ViT-base/16, 640×480 photo | top-5 labels (diff ≤ 3e-6) | 78 ms | 304 ms | 36 ms |

`bench/offline_cache.exs` compiles the DistilBERT serving into a module
cache, then runs the same code in a VM without EXLA on its code path, from
the cache only; the predictions are identical.

## Findings

  * Bumblebee servings only run fast with `preallocate_params: true`;
    otherwise the parameters are uploaded on every call (DistilBERT: 209 ms
    instead of 13 ms per call on Vulkan).
  * Image featurizers resize with NxImage `defn`s called eagerly, which use
    the global defn options. With the default evaluator a 640×480 image takes
    over 2 minutes; set `Nx.Defn.global_default_options/1` to the nx_iree
    options (and module cache on devices).
  * Text generation runs its decoding loop with host-side control, one
    synchronisation per token, so on Vulkan it is slower than on the CPU of
    this host.
  * Convolutions on the generic Vulkan target are slow compared to EXLA's CPU
    path; this is part of the Jetson performance work.
