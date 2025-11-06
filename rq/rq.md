# Embedding Generation
- [x] [Qwen 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), last token pooiling
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

- [] uniform sampling disable, 3 level, 256 each level. It would have `16M` unique combinations of semantic IDs and looks ok given 1.4M unique products and 60M unique items.
- [] uniform sampling disable, 3 level, 128 each level. It would have `2M` unique combinations of semantic IDs.
- [] uniform sampling enabled, 3 level, 256 each level. It would have `16M` unique combinations of semantic IDs.
- [] uniform sampling enabled, 3 level, 128 each level. It would have `2M` unique combinations of semantic IDs.

## RQVAE