import torch
import torch.nn as nn

class ForecastingModel(nn.Module):
    def __init__(self,
                 embedding_sizes,
                 num_numerical_features,
                 hidden_dim=32,
                 dropout=0.2):
        super(ForecastingModel, self).__init__()

        # Build embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim)
            for num_categories, emb_dim in embedding_sizes
        ])

        total_embedding_dim = sum(emb_dim for _, emb_dim in embedding_sizes)
        input_dim = total_embedding_dim + num_numerical_features

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output: single scalar (forecasted sales)
        )

    def forward(self, x_categorical, x_numerical):
        # Embeddings
        embedded = [emb_layer(x_categorical[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat(embedded + [x_numerical], dim=1)
        return self.fc(x)
