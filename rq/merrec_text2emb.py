import argparse
import collections
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from utils import *
import pandas as pd
from transformers import AutoTokenizer, AutoModel, Qwen2Model, Qwen2Tokenizer
from accelerate import Accelerator


def preprocess_text(dataset):
    print('Process text data: ')
    print(' Dataset: ', dataset)
    df = pd.read_json(path_or_buf=args.dataset, lines=True)
    print(df.head(10))
    return df

def mean_pool(last_hidden_state, attention_mask):
    masked_output = last_hidden_state * attention_mask.unsqueeze(-1)
    return masked_output.sum(dim=1) / attention_mask.sum(dim=-1, keepdim=True)

def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def generate_item_embedding(args, item_text_list, tokenizer, model, pooling, batch_size, word_drop_ratio=-1):
    print(f'Generate Text Embedding using Qwen: ')
    print(' Dataset: ', args.dataset)

    embeddings = []
    start = 0
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    processed_item = []
    with torch.no_grad():
        # while start < len(item_text_list):
        while start < 20:
            if (start+1)%100==0:
                print("==>",start+1)
            field_texts = item_text_list.iloc[start: start + batch_size]
            display(field_texts)
            sentences = field_texts["description"].tolist()
            encoded_sentences = tokenizer(sentences, max_length=args.max_sent_len,
                                            truncation=True, return_tensors='pt', padding="longest").to(args.device)
            
            # Get model outputs
            outputs = model(input_ids=encoded_sentences.input_ids,
                            attention_mask=encoded_sentences.attention_mask)
            # For Qwen models, use the last hidden state
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
            if pooling == "mean":
                mean_output = mean_pool(outputs.last_hidden_state, encoded_sentences['attention_mask'])
            elif pooling == "last":
                mean_output = last_token_pool(outputs.last_hidden_state, encoded_sentences['attention_mask'])
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
            processed_item.extend(field_texts["item_id"].tolist())
            start += batch_size

    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    np.save(args.root, embeddings)
    with open(args.root.replace("npy", "json"), "w") as fo:
        json.dump(processed_item, fo)


def load_qwen_model(model_path, device):
    """Load Qwen model and tokenizer"""
    print("Loading Qwen Model:", model_path)
    
    # Load tokenizer
    if device == torch.device("cpu"):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch.float16)
    
    # Load model
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    return tokenizer, model


def set_device(gpu_id):
    """Set device for model"""
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='provide a merrec path')
    parser.add_argument('--root', type=str, help="path to a numpy file to save the results")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='qwen')
    parser.add_argument('--plm_checkpoint', type=str,
                        default='Qwen/Qwen3-Embedding-0.6B', help='Qwen model path')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    parser.add_argument('--pooling', choices=["last", "mean"], default="last", help='pooling method to get embedding')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = set_device(args.gpu_id)
    args.device = device

    item_text_list = preprocess_text(args)

    # Load Qwen model and tokenizer
    plm_tokenizer, plm_model = load_qwen_model(args.plm_checkpoint, args.device)
    
    # Set pad token if not exists
    if plm_tokenizer.pad_token_id is None:
        if plm_tokenizer.eos_token_id is not None:
            plm_tokenizer.pad_token_id = plm_tokenizer.eos_token_id
        else:
            plm_tokenizer.pad_token_id = 0
    
    plm_model = plm_model.to(device)
    plm_model.eval()  # Set model to evaluation mode

    generate_item_embedding(args, item_text_list, plm_tokenizer,
                            plm_model, word_drop_ratio=args.word_drop_ratio, pooling=args.pooling, batch_size=args.batch_size)


