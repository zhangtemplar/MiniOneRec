# Embedding Generation
- [x] [Qwen 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), last token pooiling
- [x] [Qwen 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), last token pooiling
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
3. uniform sampling is very important to reduce collision.
4. faiss prefers 256 as codebook size than other numbers.

Industry practice for VQ-VAE and RQ-VAE suggests:
  - 1-1.5x capacity: High collision rate (10-20%)
  - 2-3x capacity: Low collision rate (3-5%) ✓ recommended
  - 5x+ capacity: Diminishing returns, wasted computation

### Qwen 0.6B

| Configurations | Codebook | Beam size | Search method | Collision Rate Before Uniform | Collision Rate After Uniform |
|----------------|----------|-----------|---------------|----------------|---------------|
| [x]            | 256*3    | 1         | def           | 0.9521         | 0.8271        |
| [x]            | 128*3    | 1         | def           | 0.9924         | 0.9649        |
| [x]            | 512*3    | 1         | def           | 0.8413         | 0.3737        |
| [x]            | 512*3    | 5         | prog          | 0.7837         | 0.4686        |
| [x]            | 256*3    | 5         | prog          | 0.9242         | 0.8456        |
| [x]            | 128*4    | 5         | prog          | 0.7385         | 0.5679        |
| [x]         | 128*8    | 5         | prog          | 0.0480         | 0.0118        |
| [x]            | 512*4    | 5         | prog          | 0.3416         | 0.1124        |
| [x]            | 1024*3   | 5         | prog          | 0.6154         | 0.2651        |
| [x]         | 256*5    | 5         | prog          | 0.2213         | 0.0970        |
| [x]         | 256*6    | 5         | prog          | 0.0988        | 0.0347        |

### Qwen 8B

| Configurations | Codebook | Beam size | Search method | Collision Rate Before Uniform | Collision Rate After Uniform |
|----------------|----------|-----------|---------------|----------------|---------------|
| [1348]         | 256*5    | 5         | prog          |          |         |
| [1349]         | 256*6    | 5         | prog          |         |         |

## RQVAE

| Parameter           | 1024-dim         | 4096-dim                   | 1024-dim                       | 4096-dim                               |
|---------------------|------------------|----------------------------|--------------------------------|----------------------------------------|
| Architecture        |                  |                            |                                |                                        |
| --layers            | 1024 512 256 128 | 4096 2048 1024 512 256 128 | 1024 512 384 256 192 128 96 64 | 4096 2048 1024 768 512 384 256 192 128 |
| --e_dim             | 64               | 128                        | 64                             | 128                                    |
| Codebook            |                  |                            |                                |                                        |
| --num_emb_list      | 512 512 512      | 512 512 512                | 256 256 256 256 256 256        | 256 256 256 256 256 256                |
| Total codes         | 134M (512³)      | 134M (512³)                |                                |                                        |
| Capacity ratio      | 2.46x            | 2.46x                      |                                |                                        |
| Training            |                  |                            |                                |                                        |
| --batch_size        | 131072 (128K)    | 65536 (64k)                | 131072 (128K)                  | 131072 (128K)                          |
| --lr                | 2e-3             | 2e-3                       | 2e-3                           | 2e-3                                   |
| --epochs            | 300              | 300                        | 300                            | 300                                    |
| --warmup_epochs     | 10               | 10                         | 10                             | 10                                     |
| --eval_step         | 10               | 10                         | 10                             | 10                                     |
| Quantization        |                  |                            |                                |                                        |
| --kmeans_init       | True             | True                       | True                           | True                                   |
| --kmeans_iters      | 100              | 100                        | 100                            | 100                                    |
| --sk_epsilons       | 0.0 0.01 0.05    | 0.0 0.01 0.05              | 0.0 0.01 0.03 0.05 0.08 0.12   | 0.0 0.01 0.03 0.05 0.08 0.12           |
| --sk_iters          | 50               | 50                         | 50                             | 50                                     |
| --beta              | 0.25             | 0.25                       | 0.25                           | 0.25                                   |
| --quant_loss_weight | 1.0              | 1.0                        | 1.0                            | 1.0                                    |
| Optimization        |                  |                            |                                |                                        |
| --learner           | AdamW            | AdamW                      | AdamW                          | AdamW                                  |
| --weight_decay      | 1e-5             | 1e-5                       | 1e-5                           | 1e-5                                   |
| --lr_scheduler_type | constant         | constant                   | constant                       | constant                               |
| Status                  | [x][]           | [x]1168 0.9397                  |   [x]0.3841 1420 | 1269 |

