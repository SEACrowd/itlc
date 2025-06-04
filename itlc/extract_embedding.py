import os
import torch
import tqdm
import numpy as np
import pandas as pd
from torch.amp import autocast
from datasets import Dataset, DatasetDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def latent_extraction(model, tokenizer, sentences, injection_layer_idx, batch_size=32, ):
    """
    Get embeddings for a list of sentences using the specified model and tokenizer.
    
    Args:
        model: The model to use for generating embeddings.
        tokenizer: The tokenizer corresponding to the model.
        sentences (list): List of sentences to encode.
        batch_size (int): Number of sentences to process in each batch.
        injection_layer_idx (int): Which layer's embeddings to return.
    
    Returns:
        torch.Tensor: Embeddings for the input sentences.
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            inputs = tokenizer(batch, padding="longest", truncation=True, return_tensors="pt", max_length=512).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            attention_mask = inputs['attention_mask']
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            

            embeddings = hidden_states[injection_layer_idx]
            
            # Mask out padding tokens
            attention_mask = attention_mask.unsqueeze(-1).expand_as(embeddings)
            masked_outputs = embeddings * attention_mask

            # Apply mean pooling over non-padding tokens
            batch_embeddings = masked_outputs.sum(dim=1) / attention_mask.sum(dim=1)

            all_embeddings.append(batch_embeddings)
            torch.cuda.empty_cache()

    return torch.cat(all_embeddings, dim=0)


def get_embeddings(model, tokenizer, language_pairs, dataset, injection_layer_idx ,batch_size=32):
    """
    Extract embeddings for a list of language pairs from the dataset.
    
    Args:
        model: The model to use for generating embeddings.
        tokenizer: The tokenizer corresponding to the model.
        language_pairs (list): List of language pairs to process.
        dataset (pd.DataFrame): Dataset containing sentences for each language pair.
    
    Returns:
        tuple: A tuple containing the embeddings and their corresponding labels.
    """
    all_embeddings = []
    all_labels = []
    
    for lang_idx, pair in enumerate(tqdm.tqdm(language_pairs, desc="Processing languages")):
        
        sentences = dataset[pair]

        if isinstance(sentences, pd.Series):
            sentences = sentences.tolist()
        elif isinstance(sentences, pd.DataFrame):
            sentences = sentences.iloc[:, 0].tolist()

        # Filter out non-string entries
        sentences = [str(s).strip() for s in sentences if isinstance(s, str) and s.strip()]

        embeddings = latent_extraction(model, tokenizer, sentences,injection_layer_idx=injection_layer_idx, batch_size=batch_size)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend([lang_idx] * len(sentences))

    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.array(all_labels)

    return final_embeddings, final_labels