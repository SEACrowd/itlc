from extract_embedding import get_embeddings
from get_language_vector import use_lda_to_get_language_vector

class ITLC:

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt, max_length=50):
        pass

    def latent_extraction(self, dataset, language_pairs, batch_size=32, layer_choice='middle_layer'):
        """
        Extract embeddings for a list of language pairs from the dataset.
        Args:
            dataset: The dataset containing the sentences.
            language_pairs: List of tuples containing language pairs.
            batch_size (int): Number of sentences to process in each batch.
            layer_choice (str): Which layer's embeddings to return ('first_layer', 'middle_layer', 'last_layer').
        Returns:
            torch.Tensor: Embeddings for the input sentences.
            list: Corresponding labels for the embeddings.
        """
        embeddings, labels = get_embeddings(self.model, self.tokenizer, language_pairs=language_pairs, dataset=dataset, 
                                            batch_size=batch_size, layer_choice=layer_choice)
        return embeddings, labels
    
    def language_vector_extraction(self, X_train, X_test, y_train, y_test, threshold=0.01, 
                                   n_components=100, n_languages=204, n_epochs=10):
        """
        Extract language vectors using LDA from the training and test data.
        Args:
            X_train: Training data embeddings.
            X_test: Test data embeddings.
            y_train: Labels for training data.
            y_test: Labels for test data.
            threshold (float): Threshold for filtering low-confidence predictions.
            n_components (int): Number of components to use in LDA.
            n_languages (int): Number of languages (classes).
            n_epochs (int): Number of epochs to train the model.
        Returns:
            language_vectors: The extracted language vectors.
        """
        language_vectors = use_lda_to_get_language_vector(X_train, X_test, y_train, y_test, threshold=threshold, 
                                                          n_components=n_components, n_languages=n_languages, 
                                                          n_epochs=n_epochs)
        return language_vectors