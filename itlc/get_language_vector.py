import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler

device='cuda' if torch.cuda.is_available() else 'cpu'

def pruning_w_lda(X_train, X_test, y_train, y_test, n_components=100, n_languages=204, n_epochs=10):
    """
    Perform LDA on the embeddings and train a simple linear model to evaluate the language vectors.
    Args:
        X_train: Training embeddings.
        X_test: Test embeddings.
        y_train: Labels for training data.
        y_test: Labels for test data.
        n_components: Number of components to use in LDA.
        n_languages: Number of languages (classes).
        n_epochs: Number of epochs to train the model.
    Returns:
        model: The trained model.
        lda: The fitted LDA model.
        X_train_lda: LDA transformed training data.
        X_test_lda: LDA transformed test data.
        y_train: Labels for training data.
        y_test: Labels for test data.
    """

    # scale the embeddings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # perform LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_train_scaled, y_train)
    X_train_lda = lda.transform(X_train_scaled)
    X_test_lda = lda.transform(X_test_scaled)

    # define a simple linear model for evaluation
    model = nn.Sequential(OrderedDict([
        ('dense', nn.Linear(n_components, n_languages)),
    ])).to(device)

    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.tensor(X_train_lda, dtype=torch.float32).to(device)
    targets = torch.tensor(y_train, dtype=torch.int64).to(device)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # train the model
    for epoch in range(n_epochs):
        model.train()
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss = loss_fn(preds, batch_targets)
            loss.backward()
            optimizer.step()

    # evaluate the model
    eval_inputs = torch.tensor(X_test_lda, dtype=torch.float32).to(device)
    eval_targets = torch.tensor(y_test, dtype=torch.int64).to(device)
    model.eval()
    with torch.no_grad():
        eval_preds = model(eval_inputs)
        _, predicted_labels = torch.max(eval_preds, 1)
        accuracy = predicted_labels.eq(eval_targets).float().mean().item()
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    return model, lda, X_train_lda, X_test_lda, y_train, y_test

def use_lda_to_get_language_vector(X_train, X_test, y_train, y_test, threshold=0.01, 
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
    # Extract the pruned weights from the model
    model, _, X_train_lda, _, y_train, _ = pruning_w_lda(X_train, X_test, y_train, y_test, 
                                                         n_components=100, n_languages=204, n_epochs=10)

    pruned_weights = model.dense.weight.data.cpu().numpy()  # [n_languages, n_components]

    # Get active dimensions based on the pruned weights
    active_dims = {}
    for lang_idx in range(n_languages):
        lang_weights = pruned_weights[lang_idx, :]  # [50]
        active_dims[lang_idx] = np.where(np.abs(lang_weights) > threshold)[0]

    # Create language vector based on the active dimension by taking the average of
    # the embedding vector on specific language
    language_vectors = {}
    for lang_idx in range(n_languages):
        active_indices = active_dims[lang_idx]
        lang_mask = y_train == lang_idx
        lang_embeds = X_train_lda[lang_mask]  # [n_samples_lang, n_components]
        language_vectors[lang_idx] = np.zeros(n_components)  # [n_components]

        if len(active_indices) > 0 and lang_embeds.shape[0] > 0:
            lang_embeds_active = lang_embeds[:, active_indices]  # [n_samples_lang, n_active]
            active_means = np.mean(lang_embeds_active, axis=0)  # [n_active]
            language_vectors[lang_idx][active_indices] = active_means
    
    return language_vectors