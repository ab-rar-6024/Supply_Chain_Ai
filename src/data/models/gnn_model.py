import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

# Simplified GNN (works without torch-geometric)
class SimplifiedRiskGNN(nn.Module):
    """Simplified GNN that works without torch-geometric"""
    
    def __init__(self, node_features: Dict[str, int], hidden_channels: int = 64):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Encoders for different node types
        self.encoders = nn.ModuleDict({
            'supplier': nn.Linear(4, hidden_channels),
            'warehouse': nn.Linear(3, hidden_channels),
            'retailer': nn.Linear(3, hidden_channels)
        })
        
        # Graph convolution layers (using adjacency matrix)
        self.conv1 = nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)
        
        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, node_features: Dict[str, torch.Tensor], 
                adjacency: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass using adjacency matrices
        """
        
        # Encode nodes
        x_dict = {}
        for node_type, encoder in self.encoders.items():
            if node_type in node_features:
                x_dict[node_type] = encoder(node_features[node_type])
            else:
                x_dict[node_type] = torch.zeros(1, self.hidden_channels)
        
        # Message passing
        for _ in range(2):  # 2 layers
            new_x_dict = {}
            
            # Process each edge type
            for edge_type, adj in adjacency.items():
                src_type, dst_type = edge_type.split('->')
                
                if src_type in x_dict and dst_type in x_dict:
                    # Message passing: A * X
                    messages = torch.mm(adj, x_dict[src_type])
                    
                    # Combine with current features
                    if dst_type not in new_x_dict:
                        new_x_dict[dst_type] = messages
                    else:
                        new_x_dict[dst_type] += messages
            
            # Update node features
            for node_type in x_dict:
                if node_type in new_x_dict:
                    x_dict[node_type] = F.relu(self.conv1(x_dict[node_type] + new_x_dict[node_type]))
                else:
                    x_dict[node_type] = F.relu(self.conv1(x_dict[node_type]))
                    
                x_dict[node_type] = self.dropout(x_dict[node_type])
        
        # Predict risk scores
        risk_scores = {}
        for node_type, features in x_dict.items():
            risk_scores[node_type] = self.risk_head(features).squeeze()
            
        return risk_scores
    
    def predict_risk_propagation(self, node_features: Dict, adjacency: Dict, 
                                 initial_risk_nodes: Dict) -> Dict:
        """Predict risk propagation"""
        self.eval()
        with torch.no_grad():
            risk_scores = self.forward(node_features, adjacency)
            
            # Apply initial risk boost
            for node_type, nodes in initial_risk_nodes.items():
                if node_type in risk_scores:
                    for node_idx in nodes:
                        if node_idx < len(risk_scores[node_type]):
                            risk_scores[node_type][node_idx] = torch.clamp(
                                risk_scores[node_type][node_idx] + 0.3, 0, 1
                            )
                            
        return risk_scores

# Original GNN (requires torch-geometric)
class RiskPropagationGNN(nn.Module):
    """Original GNN with torch-geometric support"""
    
    def __init__(self, hidden_channels: int = 64, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Node type specific encoders
        self.supplier_encoder = nn.Linear(4, hidden_channels)
        self.warehouse_encoder = nn.Linear(3, hidden_channels)
        self.retailer_encoder = nn.Linear(3, hidden_channels)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hetero_data):
        """Forward pass (simplified for compatibility)"""
        # Encode nodes
        x_dict = {
            'supplier': self.supplier_encoder(hetero_data['supplier'].x),
            'warehouse': self.warehouse_encoder(hetero_data['warehouse'].x),
            'retailer': self.retailer_encoder(hetero_data['retailer'].x)
        }
        
        # Predict risk scores
        risk_scores = {}
        for node_type, features in x_dict.items():
            risk_scores[node_type] = self.output_layer(features).squeeze()
            
        return risk_scores
    
    def predict_risk_propagation(self, hetero_data, initial_risk_nodes):
        """Predict risk propagation"""
        self.eval()
        with torch.no_grad():
            risk_scores = self.forward(hetero_data)
            
            # Apply initial risk
            for node_type, nodes in initial_risk_nodes.items():
                if node_type in risk_scores:
                    for node_idx in nodes:
                        if node_idx < len(risk_scores[node_type]):
                            risk_scores[node_type][node_idx] = torch.clamp(
                                risk_scores[node_type][node_idx] + 0.3, 0, 1
                            )
        return risk_scores

class TemporalGNN(nn.Module):
    """Temporal GNN placeholder"""
    def __init__(self, hidden_channels: int = 64, num_timesteps: int = 30):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        
    def forward(self, temporal_graphs: list):
        return torch.randn(1, self.hidden_channels)