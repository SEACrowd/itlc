import os
import torch
import tqdm
import numpy as np
import pandas as pd
from torch.amp import autocast

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def latent_extraction(model, tokenizer, sentences, batch_size=32, layer_choice='middle_layer'):
    """
    Get embeddings for a list of sentences using the specified model and tokenizer.
    
    Args:
        model: The model to use for generating embeddings.
        tokenizer: The tokenizer corresponding to the model.
        sentences (list): List of sentences to encode.
        batch_size (int): Number of sentences to process in each batch.
        layer_choice (str): Which layer's embeddings to return ('first_layer', 'middle_layer', 'last_layer').
    
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
            
            num_layers = len(hidden_states) - 1 # Excluding the initial embedding layer
            if layer_choice == 'first_layer':
                embeddings = hidden_states[1]
            elif layer_choice == 'middle_layer':
                mid_index = num_layers // 2
                embeddings = hidden_states[mid_index]
            elif layer_choice == 'last_layer':
                embeddings = hidden_states[-1]
            else:
                raise ValueError("Invalid layer choice. Choose from 'first_layer', 'middle_layer', or 'last_layer'.")
            
            # Mask out padding tokens
            attention_mask = attention_mask.unsqueeze(-1).expand_as(embeddings)
            masked_outputs = embeddings * attention_mask

            # Apply mean pooling over non-padding tokens
            batch_embeddings = masked_outputs.sum(dim=1) / attention_mask.sum(dim=1)

            embeddings.append(batch_embeddings)
            torch.cuda.empty_cache()

    return torch.cat(all_embeddings, dim=0)


def get_embeddings(model, tokenizer, language_pairs, dataset, batch_size=32, layer_choice='middle_layer'):
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

    # if hugginfaceface dataset is detected, convert to pandas Series or DataFrame
    if isinstance(dataset, pd.Series):
        dataset = dataset.to_frame()
    elif isinstance(dataset, pd.DataFrame):
        # Ensure the DataFrame has a single column for each language pair
        if dataset.shape[1] > 1:
            dataset = dataset.iloc[:, 0]
    elif not isinstance(dataset, pd.DataFrame):
        raise ValueError("Dataset must be a pandas Series or DataFrame.")
    
    for lang_idx, pair in enumerate(tqdm.tqdm(language_pairs, desc="Processing languages")):
        
        sentences = dataset[pair]

        if isinstance(sentences, pd.Series):
            sentences = sentences.tolist()
        elif isinstance(sentences, pd.DataFrame):
            sentences = sentences.iloc[:, 0].tolist()

        # Filter out non-string entries
        sentences = [str(s).strip() for s in sentences if isinstance(s, str) and s.strip()]

        embeddings = latent_extraction(model, tokenizer, sentences, batch_size=batch_size, layer_choice=layer_choice)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend([lang_idx] * len(sentences))

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)

    return embeddings, labels