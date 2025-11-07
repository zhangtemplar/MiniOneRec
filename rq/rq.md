# Embedding Generation
- [x] [Qwen 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), last token pooiling
- [1034] [Qwen 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), last token pooiling
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

Industry practice for VQ-VAE and RQ-VAE suggests:
  - 1-1.5x capacity: High collision rate (10-20%)
  - 2-3x capacity: Low collision rate (3-5%) ✓ recommended
  - 5x+ capacity: Diminishing returns, wasted computation

- [x] uniform sampling disable, 3 level, 256 each level. It would have `16M` unique combinations of semantic IDs and looks ok given 1.4M unique products and 60M unique items.
- [x] uniform sampling enabled, 3 level, 256 each level. It would have `16M` unique combinations of semantic IDs.
- [x] uniform sampling disable, 3 level, 128 each level. It would have `2M` unique combinations of semantic IDs.
- [x] uniform sampling enabled, 3 level, 128 each level. It would have `2M` unique combinations of semantic IDs.
- [x] uniform sampling disable, 3 level, 512 each level. It would have `134M` unique combinations of semantic IDs.
- [x] uniform sampling enabled, 3 level, 512 each level. It would have `134M` unique combinations of semantic IDs.

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


### 512x512x512
Before balancing:
  total=54513856
  L1: unique=256
  L2: unique=256
  L3: unique=256
  L4: unique=8
  unique full-paths=8653621  collision_rate=0.8413
Saved indices: /mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_train_512_512_512.json
Saved faiss quantizer: /mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_train_512_512_512.faiss
loading test: /mnt/lustre/metavmds0lstre/data/rankagi/external_dataset/minionerec/rankagi_output_v2_item_text_train.npy
0
shape: (9620093, 1024)
0
Encoding 9620093 vectors ...
  done, codes.shape=(9620093, 4)

0
Before balancing:
  total=9620093
  L1: unique=256
  L2: unique=256
  L3: unique=256
  L4: unique=8
  unique full-paths=3538640  collision_rate=0.6322


## RQVAE

  | Parameter               | 1024-dim         | 4096-dim              | 4096-dim                   |
  |-------------------------|------------------|-----------------------|----------------------------|
  | Architecture            |                  |                       |                            |
  | --layers                | 1024 512 256 128 | 4096 2048 1024 512 256 128 | 4096 2048 1024 512 256 128 |
  | --e_dim                 | 64               | 128                    | 128                        |
  | Codebook                |                  |                       |                            |
  | --num_emb_list          | 512 512 512      | 512 512 512           | 512 512 512 512            |
  | Total codes             | 134M (512³)      | 134M (512³)           | 68.7B (512⁴)               |
  | Capacity ratio          | 2.46x            | 2.46x                 | 1,260x                     |
  | Training                |                  |                       |                            |
  | --batch_size            | 131072 (128K)    | 65536 (64k)           | 131072 (128K)              |
  | --lr                    | 2e-3             | 2e-3                  | 2e-3                       |
  | --epochs                | 300              | 300                   | 300                        |
  | --warmup_epochs         | 10               | 10                    | 10                         |
  | --eval_step             | 10               | 10                    | 10                         |
  | Quantization            |                  |                       |                            |
  | --kmeans_init           | True             | True                  | True                       |
  | --kmeans_iters          | 100              | 100                   | 100                        |
  | --sk_epsilons           | 0.0 0.01 0.05    | 0.0 0.01 0.05         | 0.0 0.01 0.05 0.1          |
  | --sk_iters              | 50               | 50                    | 50                         |
  | --beta                  | 0.25             | 0.25                  | 0.25                       |
  | --quant_loss_weight     | 1.0              | 1.0                   | 1.0                        |
  | Optimization            |                  |                       |                            |
  | --learner               | AdamW            | AdamW                 | AdamW                      |
  | --weight_decay          | 1e-5             | 1e-5                  | 1e-5                       |
  | --lr_scheduler_type     | constant         | constant              | constant                   |
  | System                  |                  |                       |                            |
  | --num_workers           | 16               | 16                    | 16                         |
  | --device                | cuda:0           | cuda:0                | cuda:0                     |
  | Performance Estimates   |                  |                       |                            |
  | GPU memory usage        | ~15 GB (9%)      | ~35 GB (20%)          | ~45 GB (26%)               |
  | Batches per epoch       | 416              | 555                   | 416                        |
  | Time per epoch          | 3-5 min          | 5-8 min               | 5-8 min                    |
  | Total training time     | 15-25 hours      | 25-40 hours           | 25-40 hours                |
  | Samples per code        | 256              | 192                   | 256                        |
  | Expected collision rate | 2-3%             | 2-3%                  | <1%                        |
  | Expected MSE            | <0.005           | <0.005                | <0.003                     |
  |-------------------------|------------------|-----------------------|----------------------------|
  | Status                  | 1038             | TBD                   |                            |

