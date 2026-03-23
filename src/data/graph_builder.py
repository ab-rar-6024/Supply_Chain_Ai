import torch
import numpy as np
from typing import Dict, List, Tuple

class SupplyChainGraphBuilder:
    """Builds and manages the supply chain graph structure"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def build_graph(self):
        """Build graph - returns a dictionary with node features and adjacency"""
        num_suppliers = self.config['supply_chain_config']['num_suppliers']
        num_warehouses = self.config['supply_chain_config']['num_warehouses']
        num_retailers = self.config['supply_chain_config']['num_retailers']
        
        # Node features
        node_features = {
            'supplier': self._generate_supplier_features(num_suppliers),
            'warehouse': self._generate_warehouse_features(num_warehouses),
            'retailer': self._generate_retailer_features(num_retailers)
        }
        
        # Adjacency matrices
        adjacency = {
            'supplier->warehouse': self._create_supplier_warehouse_adjacency(num_suppliers, num_warehouses),
            'warehouse->retailer': self._create_warehouse_retailer_adjacency(num_warehouses, num_retailers)
        }
        
        return {
            'node_features': node_features,
            'adjacency': adjacency
        }
    
    def _generate_supplier_features(self, n: int) -> torch.Tensor:
        features = []
        for _ in range(n):
            reliability = np.random.beta(2, 5)
            capacity = np.random.lognormal(mean=8, sigma=1)
            cost = np.random.normal(100, 20)
            location_risk = np.random.uniform(0, 1)
            features.append([reliability, capacity, cost, location_risk])
        return torch.tensor(features, dtype=torch.float)
    
    def _generate_warehouse_features(self, n: int) -> torch.Tensor:
        features = []
        for _ in range(n):
            capacity = np.random.lognormal(mean=10, sigma=1)
            storage_cost = np.random.normal(50, 10)
            automation_level = np.random.uniform(0, 1)
            features.append([capacity, storage_cost, automation_level])
        return torch.tensor(features, dtype=torch.float)
    
    def _generate_retailer_features(self, n: int) -> torch.Tensor:
        features = []
        for _ in range(n):
            demand_rate = np.random.gamma(shape=2, scale=10)
            inventory_level = np.random.poisson(lam=100)
            customer_importance = np.random.uniform(0, 1)
            features.append([demand_rate, inventory_level, customer_importance])
        return torch.tensor(features, dtype=torch.float)
    
    def _create_supplier_warehouse_adjacency(self, num_suppliers: int, num_warehouses: int) -> torch.Tensor:
        adj = torch.zeros(num_warehouses, num_suppliers)
        for supplier in range(num_suppliers):
            num_connections = np.random.randint(2, min(6, num_warehouses + 1))
            warehouses = np.random.choice(num_warehouses, num_connections, replace=False)
            for warehouse in warehouses:
                adj[warehouse, supplier] = 1.0
        return adj
    
    def _create_warehouse_retailer_adjacency(self, num_warehouses: int, num_retailers: int) -> torch.Tensor:
        adj = torch.zeros(num_retailers, num_warehouses)
        for warehouse in range(num_warehouses):
            num_connections = np.random.randint(5, min(21, num_retailers + 1))
            retailers = np.random.choice(num_retailers, num_connections, replace=False)
            for retailer in retailers:
                adj[retailer, warehouse] = 1.0
        return adj