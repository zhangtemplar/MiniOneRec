# Embedding Generation
- [x] [Qwen 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), last token pooiling
- [] [Qwen 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), last token pooiling
- [] [Gemmea 3 270M](https://huggingface.co/google/gemma-3-270m)

There are `54513856` items in training split and `9620093` items in testing split. `1403098` unique product id (brand x c2 category)

# Tokenization
## RQ Kmeans
The supported parameters:
1. uniform sampling (default false)
2. codebook level (default 3)
3. codebook size per level (default 256)

Note:
1. ResidualQuantizer doesn't support different codebooks across levels.
2. GCP server has 4TB CPU memory and 8 B200 GPUs with 183GB GPU memory. The job needs at least 500GB memory.

- [x] uniform sampling disable, 3 level, 256 each level. It would have `16M` unique combinations of semantic IDs and looks ok given 1.4M unique products and 60M unique items.
- [x] uniform sampling disable, 3 level, 128 each level. It would have `2M` unique combinations of semantic IDs.
- [x] uniform sampling enabled, 3 level, 256 each level. It would have `16M` unique combinations of semantic IDs.
- [x] uniform sampling enabled, 3 level, 128 each level. It would have `2M` unique combinations of semantic IDs.

### 128x128x128
```
Before balancing:
  total=54513856
  L1: unique=255
  L2: unique=256
  L3: unique=32
  unique full-paths=414549  collision_rate=0.9924

=== Sinkhorn uniform mapping  level 1/3 ===
  Sinkhorn level: N=54513856  K=128  tau=693.39  iters=30  batch=8192
    level balanced: min=259616  max=725972

=== Sinkhorn uniform mapping  level 2/3 ===
  Sinkhorn level: N=54513856  K=128  tau=602.35  iters=30  batch=8192
    level balanced: min=337999  max=624244

=== Sinkhorn uniform mapping  level 3/3 ===
  Sinkhorn level: N=54513856  K=128  tau=447.4  iters=30  batch=8192
    level balanced: min=335350  max=537411
0
After  balancing:
  total=54513856
  L1: unique=128
  L2: unique=128
  L3: unique=128
  unique full-paths=1914213  collision_rate=0.9649
```

### 256x256x256
Before balancing:
  total=54513856
  L1: unique=256
  L2: unique=256
  L3: unique=256
  unique full-paths=2609119  collision_rate=0.9521

=== Sinkhorn uniform mapping  level 1/3 ===
  Sinkhorn level: N=54513856  K=256  tau=742.75  iters=30  batch=8192
    level balanced: min=122162  max=842311

=== Sinkhorn uniform mapping  level 2/3 ===
  Sinkhorn level: N=54513856  K=256  tau=621.73  iters=30  batch=8192
    level balanced: min=126322  max=492693

=== Sinkhorn uniform mapping  level 3/3 ===
  Sinkhorn level: N=54513856  K=256  tau=384.07  iters=30  batch=8192
    level balanced: min=106125  max=601047
0
After  balancing:
  total=54513856
  L1: unique=256
  L2: unique=256
  L3: unique=256
  unique full-paths=9426059  collision_rate=0.8271

## RQVAE