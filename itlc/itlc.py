import os
import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .extract_embedding import get_embeddings
from .get_language_vector import use_lda_to_get_language_vector


class ITLC:
    """
    A unified ITLC class that provides:
      1) latent_extraction(...) via get_embeddings
      2) language_vector_extraction(...) via LDA (and saves out the LDA model + lang vectors)
      3) generate(...) which injects LDA-based shift vectors in the middle layer.

    If `lda_model_path` or `langvec_path` are provided (and point to existing files),
    they will be loaded in __init__. Otherwise, you must call `language_vector_extraction`
    before calling `generate`.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device = None,
        injection_layer_idx: int = None,
        lda_model_path: str = None,
        langvec_path: str = None,
        n_components:int = 100,
    ):
        # 0. Set default device if not provided
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # 1. Store the HF model, tokenizer, and device
        self.model = model.to(device)
        self.tokenizer = tokenizer

        # 2. If no layer was specified, default to the “middle” layer
        if injection_layer_idx is None:
            # config.num_hidden_layers is available after you do model.to(device)
            self.injection_layer_idx = self.model.config.num_hidden_layers // 2
        else:
            self.injection_layer_idx = injection_layer_idx

        # 3. Initialize placeholders for LDA pseudoinverse and language-vectors
        self.lda_pinv = None
        self.language_vectors = None
        self.n_components = n_components

        # 4. Attempt to load precomputed LDA and language_vectors if paths are provided
        self.lda_model_path = lda_model_path
        self.langvec_path = langvec_path

        if lda_model_path and os.path.exists(lda_model_path):
            lda = joblib.load(lda_model_path)
            lda_scalings = lda.scalings_[:, :self.n_components] 
            # compute pseudoinverse: 
            self.lda_pinv = torch.tensor(
                np.linalg.pinv(lda_scalings),
                dtype=torch.float32,
                device=self.model.device,
            )
        else:
            # no LDA loaded yet
            self.lda_pinv = None

        if langvec_path and os.path.exists(langvec_path):
            raw_langvec = joblib.load(langvec_path)
            self.language_vectors = {
                lang_id: torch.tensor(vec, dtype=torch.float32)
                for lang_id, vec in raw_langvec.items()
            }
        else:
            # no language_vectors loaded yet
            self.language_vectors = None

        # model in eval mode
        self.model.eval()

    def generate(
        self,
        prompt: list,
        src_id: int,
        tgt_id: int,
        scale: float = 0.5,
        shift_strategy: str = "prompt_and_gen",     # "prompt_only", "gen_only", "prompt_and_gen"
        task: str = "crosslingual",       # "crosslingual" or "monolingual"
        do_sample: bool =True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.5,
        **extra_kwargs
    ) -> str:
        """
        Perform controlled generation on a single prompt by injecting the LDA-computed shift
        vector at layer `self.injection_layer_idx`.

        Before calling generate, ensure that `self.lda_pinv` and `self.language_vectors` are loaded.
        If not, call `language_vector_extraction(...)` first.

        Args:
            prompt (str): A single-sentence prompt (e.g. "Translate the following ...").
            source_lang_code (str): Two-letter code for the source language (e.g. "en").
            target_lang_code (str): Two-letter code for the target language (e.g. "id").
            max_new_tokens (int): How many tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int): Top-k sampling.
            top_p (float): Top-p sampling.
            **extra_kwargs: Any other kwargs for HF’s generate (e.g. repetition_penalty).

        Returns:
            str: The decoded generated text (no special tokens).
        """
        # Ensure LDA and language vectors are available
        if self.lda_pinv is None or self.language_vectors is None:
            raise RuntimeError(
                "LDA or language_vectors not loaded. "
                "Call `language_vector_extraction(...)` first or supply valid paths to load in __init__."
            )

        # Retrieve the n-dim vectors, move to device, project to ori-dim vector
        src_vec = self.language_vectors[src_id].to(self.model.device)
        tgt_vec = self.language_vectors[tgt_id].to(self.model.device)

        src_vec_ori = src_vec @ self.lda_pinv
        tgt_vec_ori = tgt_vec @ self.lda_pinv

        # C) Static scaling

        scale_sub = scale
        scale_add = scale

        # D) Build the shift vector in 896 dims
        if task == "crosslingual":
            shift_vec = -src_vec_ori * scale_sub + tgt_vec_ori * scale_add  # (896,)
        elif task == "monolingual":
            shift_vec = tgt_vec_ori * scale_add
        else:
            raise ValueError(f"Unknown task: {task!r}")

        # E) Tokenize the prompt (batch size = 1)
        batch = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        input_ids = batch.input_ids          # (1, seq_len)
        attention_mask = batch.attention_mask   # (1, seq_len)

        # F) Register a forward-hook on the chosen transformer block
        handle = None
        block_idx = self.injection_layer_idx - 1
        block = self.model.model.layers[block_idx]

        def hook_fn(module, inputs, outputs):
            """
            Adds shift_vec to hidden_states depending on seq_len_cur and shift_strategy.
            """
            hidden_states = outputs[0]  # (B, seq_len_cur, hidden_dim)
            B, seq_len_cur, hidden_dim = hidden_states.size()

            dyn_shift = shift_vec.view(1, 1, hidden_dim).expand(B, seq_len_cur, hidden_dim).to(hidden_states.device)
            # If encoding prompt (seq_len_cur > 1), apply only where attn=1
            if seq_len_cur > 1 and shift_strategy in ("prompt_only", "prompt_and_gen"):
                attn = attention_mask.unsqueeze(-1).type_as(dyn_shift)  # (1, seq_len_cur, 1)
                hidden_states = hidden_states + dyn_shift * attn

            # If generating new token (seq_len_cur == 1), apply to whole vector
            if seq_len_cur == 1 and shift_strategy in ("gen_only", "prompt_and_gen"):
                hidden_states = hidden_states + dyn_shift

            return (hidden_states, *outputs[1:])

        handle = block.register_forward_hook(hook_fn)

        # G) Call HuggingFace’s generate
        output_ids = self.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **extra_kwargs,
        )

        # H) Remove the hook to avoid side effects
        if handle is not None:
            handle.remove()

        # I) Decode only the newly generated tokens
        gen_ids = output_ids[:, input_ids.shape[1]:]  # drop prompt tokens
        generated_text = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return generated_text

    def latent_extraction(
        self,
        dataset,
        language_pairs,
        batch_size: int = 32,
    ):
        """
        Extract embeddings for a list of language pairs from the dataset.

        Args:
            dataset: The dataset containing the sentences.
            language_pairs: List of tuples of language codes to extract.
            batch_size (int): Number of sentences per batch.
            layer_choice (str): Which layer’s embeddings ('first_layer',
                                'middle_layer', 'last_layer').

        Returns:
            embeddings (torch.Tensor): The extracted embeddings.
            labels (list): Corresponding labels.
        """
            
        embeddings, labels = get_embeddings(
            self.model,
            self.tokenizer,
            language_pairs=language_pairs,
            dataset=dataset,
            batch_size=batch_size,
            injection_layer_idx=self.injection_layer_idx,
        )
        return embeddings, labels

    def language_vector_extraction(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        threshold: float = 0.01,
        n_languages: int = 204,
        n_epochs: int = 10,
    ):
        """
        Extract language vectors via LDA, then save both the LDA model and the language_vectors dict.

        After calling this method, `self.lda_pinv` and `self.language_vectors` will be populated
        so that you can immediately call `generate(...)`.

        Args:
            X_train: Training embeddings (from latent_extraction).
            X_test: Test embeddings.
            y_train: Labels for X_train.
            y_test: Labels for X_test.
            lda_model_save_path (str): Path to save the trained LDA model (.pkl).
            langvec_save_path (str): Path to save the language_vectors dict (.pkl).
            threshold (float): Threshold for filtering (passed to use_lda_to_get_language_vector).
            n_languages (int): Number of distinct languages/classes.
            n_epochs (int): Number of epochs to fit LDA.

        Returns:
            lda_model: The trained LDA model.
            language_vectors: A dict[int → 100-dim np.ndarray] of language vectors.
        """
        # 1. Run LDA-based language-vector extraction
        lda_model, language_vectors = use_lda_to_get_language_vector(
            X_train,
            X_test,
            y_train,
            y_test,
            threshold=threshold,
            n_components=self.n_components,
            n_languages=n_languages,
            n_epochs=n_epochs,
        )

        return lda_model, language_vectors
