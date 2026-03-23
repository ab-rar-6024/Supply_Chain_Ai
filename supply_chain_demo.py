"""
Supply Chain AI System - Single File Demo (Fixed)
Run with: python supply_chain_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import yaml
import logging
import warnings
from tqdm import tqdm
from typing import Dict, List, Tuple
import random
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# GRAPH BUILDER
# ============================================================================

class SupplyChainGraphBuilder:
    """Builds and manages the supply chain graph structure"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.num_suppliers = config['supply_chain_config']['num_suppliers']
        self.num_warehouses = config['supply_chain_config']['num_warehouses']
        self.num_retailers = config['supply_chain_config']['num_retailers']
        
    def build_graph(self):
        """Build graph - returns node features and adjacency matrices"""
        
        # Node features
        node_features = {
            'supplier': self._generate_supplier_features(),
            'warehouse': self._generate_warehouse_features(),
            'retailer': self._generate_retailer_features()
        }
        
        # Adjacency matrices
        adjacency = {
            'supplier->warehouse': self._create_supplier_warehouse_adjacency(),
            'warehouse->retailer': self._create_warehouse_retailer_adjacency()
        }
        
        return node_features, adjacency
    
    def _generate_supplier_features(self):
        """Generate supplier features"""
        features = []
        for i in range(self.num_suppliers):
            # Make some suppliers inherently more risky
            reliability = np.random.beta(2, 5)
            if i % 5 == 0:  # Every 5th supplier is less reliable
                reliability = np.random.beta(1, 3)
            capacity = np.random.lognormal(mean=8, sigma=1)
            cost = np.random.normal(100, 20)
            location_risk = np.random.uniform(0, 1)
            features.append([reliability, capacity, cost, location_risk])
        return torch.tensor(features, dtype=torch.float)
    
    def _generate_warehouse_features(self):
        """Generate warehouse features"""
        features = []
        for _ in range(self.num_warehouses):
            capacity = np.random.lognormal(mean=10, sigma=1)
            storage_cost = np.random.normal(50, 10)
            automation_level = np.random.uniform(0, 1)
            features.append([capacity, storage_cost, automation_level])
        return torch.tensor(features, dtype=torch.float)
    
    def _generate_retailer_features(self):
        """Generate retailer features"""
        features = []
        for _ in range(self.num_retailers):
            demand_rate = np.random.gamma(shape=2, scale=10)
            inventory_level = np.random.poisson(lam=100)
            customer_importance = np.random.uniform(0, 1)
            features.append([demand_rate, inventory_level, customer_importance])
        return torch.tensor(features, dtype=torch.float)
    
    def _create_supplier_warehouse_adjacency(self):
        """Create adjacency matrix for supplier->warehouse connections"""
        adj = torch.zeros(self.num_warehouses, self.num_suppliers)
        for supplier in range(self.num_suppliers):
            num_connections = np.random.randint(2, min(6, self.num_warehouses + 1))
            warehouses = np.random.choice(self.num_warehouses, num_connections, replace=False)
            for warehouse in warehouses:
                adj[warehouse, supplier] = 1.0
        return adj
    
    def _create_warehouse_retailer_adjacency(self):
        """Create adjacency matrix for warehouse->retailer connections"""
        adj = torch.zeros(self.num_retailers, self.num_warehouses)
        for warehouse in range(self.num_warehouses):
            num_connections = np.random.randint(5, min(21, self.num_retailers + 1))
            retailers = np.random.choice(self.num_retailers, num_connections, replace=False)
            for retailer in retailers:
                adj[retailer, warehouse] = 1.0
        return adj

# ============================================================================
# GNN MODEL
# ============================================================================

class SimplifiedRiskGNN(nn.Module):
    """Simplified GNN for risk propagation"""
    
    def __init__(self, hidden_channels: int = 64):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        
        # Encoders for different node types
        self.supplier_encoder = nn.Linear(4, hidden_channels)
        self.warehouse_encoder = nn.Linear(3, hidden_channels)
        self.retailer_encoder = nn.Linear(3, hidden_channels)
        
        # Graph convolution layers
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
        
    def forward(self, node_features: Dict, adjacency: Dict) -> Dict:
        """Forward pass"""
        
        # Encode nodes
        x_dict = {
            'supplier': self.supplier_encoder(node_features['supplier']),
            'warehouse': self.warehouse_encoder(node_features['warehouse']),
            'retailer': self.retailer_encoder(node_features['retailer'])
        }
        
        # Message passing
        for _ in range(2):
            new_x_dict = {}
            
            # Supplier -> Warehouse
            adj_sw = adjacency.get('supplier->warehouse')
            if adj_sw is not None and adj_sw.shape[1] == x_dict['supplier'].shape[0]:
                messages_sw = torch.mm(adj_sw, x_dict['supplier'])
                new_x_dict['warehouse'] = messages_sw
            
            # Warehouse -> Retailer
            adj_wr = adjacency.get('warehouse->retailer')
            if adj_wr is not None and adj_wr.shape[1] == x_dict['warehouse'].shape[0]:
                messages_wr = torch.mm(adj_wr, x_dict['warehouse'])
                new_x_dict['retailer'] = messages_wr
            
            # Update features
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
    
    def predict_risk_propagation(self, node_features, adjacency, initial_risks):
        """Predict risk propagation with initial risks"""
        self.eval()
        with torch.no_grad():
            risk_scores = self.forward(node_features, adjacency)
            
            # Apply initial risk boost
            for node_type, nodes in initial_risks.items():
                if node_type in risk_scores and len(risk_scores[node_type]) > 0:
                    for node_idx in nodes:
                        if node_idx < len(risk_scores[node_type]):
                            # Boost risk for affected nodes
                            risk_scores[node_type][node_idx] = torch.clamp(
                                risk_scores[node_type][node_idx] + 0.5, 0, 1
                            )
                            
                            # Propagate risk to connected nodes
                            if node_type == 'supplier':
                                # Find connected warehouses
                                adj_sw = adjacency.get('supplier->warehouse')
                                if adj_sw is not None:
                                    connected_warehouses = torch.where(adj_sw[:, node_idx] > 0)[0]
                                    for wh in connected_warehouses:
                                        if wh < len(risk_scores['warehouse']):
                                            risk_scores['warehouse'][wh] = torch.clamp(
                                                risk_scores['warehouse'][wh] + 0.3, 0, 1
                                            )
                                            
                                            # Find connected retailers
                                            adj_wr = adjacency.get('warehouse->retailer')
                                            if adj_wr is not None:
                                                connected_retailers = torch.where(adj_wr[:, wh] > 0)[0]
                                                for ret in connected_retailers:
                                                    if ret < len(risk_scores['retailer']):
                                                        risk_scores['retailer'][ret] = torch.clamp(
                                                            risk_scores['retailer'][ret] + 0.2, 0, 1
                                                        )
        return risk_scores

# ============================================================================
# RL ENVIRONMENT
# ============================================================================

class SupplyChainEnv:
    """Custom environment for supply chain simulation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.current_day = 0
        self.reset()
        
    def reset(self):
        """Reset environment"""
        self.current_day = 0
        self.inventory = self._initialize_inventory()
        self.costs = []
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        inventory_adj = action[1] if len(action) > 1 else 0
        
        # Update inventory
        for warehouse in self.inventory['warehouse']:
            new_inv = warehouse['current'] + inventory_adj
            warehouse['current'] = np.clip(new_inv, 0, warehouse['max'])
        
        # Calculate costs
        holding_cost = self._calculate_holding_cost()
        penalty_cost = self._calculate_penalty_cost()
        total_cost = holding_cost + penalty_cost
        self.costs.append(total_cost)
        
        # Calculate reward
        reward = -total_cost + self._calculate_service_level()
        
        # Check if done
        self.current_day += 1
        done = self.current_day >= self.config['supply_chain_config']['simulation_days']
        
        return self._get_observation(), reward, done
    
    def _get_observation(self):
        """Get current observation"""
        obs = []
        for warehouse in self.inventory['warehouse']:
            obs.extend([warehouse['current'], warehouse['max'], warehouse['reorder_point']])
        for retailer in self.inventory['retailer']:
            obs.extend([retailer['current'], retailer['demand']])
        
        # Pad to fixed size
        expected_size = 100
        if len(obs) < expected_size:
            obs.extend([0] * (expected_size - len(obs)))
        return np.array(obs[:expected_size], dtype=np.float32)
    
    def _initialize_inventory(self):
        """Initialize inventory"""
        config = self.config['supply_chain_config']
        
        warehouse_inventory = []
        for _ in range(config['num_warehouses']):
            warehouse_inventory.append({
                'current': np.random.poisson(200),
                'max': 500,
                'reorder_point': 100
            })
            
        retailer_inventory = []
        for _ in range(config['num_retailers']):
            retailer_inventory.append({
                'current': np.random.poisson(50),
                'demand': np.random.gamma(2, 10)
            })
            
        return {
            'warehouse': warehouse_inventory,
            'retailer': retailer_inventory
        }
    
    def _calculate_holding_cost(self):
        """Calculate holding cost"""
        total = sum(w['current'] for w in self.inventory['warehouse'])
        total += sum(r['current'] for r in self.inventory['retailer'])
        return total * 0.01
    
    def _calculate_penalty_cost(self):
        """Calculate stockout penalty"""
        penalty = 0
        for retailer in self.inventory['retailer']:
            if retailer['current'] < retailer['demand']:
                penalty += (retailer['demand'] - retailer['current']) * 10
        return penalty
    
    def _calculate_service_level(self):
        """Calculate service level"""
        total_demand = sum(r['demand'] for r in self.inventory['retailer'])
        total_available = sum(r['current'] for r in self.inventory['retailer'])
        if total_demand > 0:
            return min(1.0, total_available / total_demand) * 100
        return 100

class SimpleRLOptimizer:
    """Simple rule-based optimizer"""
    
    def __init__(self, env):
        self.env = env
        
    def get_action(self, observation):
        """Simple rule-based action"""
        # Check if inventory is low (simple heuristic)
        if len(observation) > 0:
            avg_inventory = np.mean(observation[:len(self.env.inventory['warehouse'])])
            if avg_inventory < 100:
                return np.array([0.5, 50, 0.5])  # Increase inventory
            elif avg_inventory > 400:
                return np.array([0.5, -30, 0.5])  # Decrease inventory
        return np.array([0.5, 0, 0.5])  # Do nothing

# ============================================================================
# MAIN SYSTEM
# ============================================================================

class SupplyChainAI:
    """Main Supply Chain AI System"""
    
    def __init__(self, config_path='config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all components"""
        print("\n" + "="*60)
        print("Initializing Supply Chain AI System...")
        print("="*60)
        
        # Graph Builder
        self.graph_builder = SupplyChainGraphBuilder(self.config)
        self.node_features, self.adjacency = self.graph_builder.build_graph()
        print(f"✓ Graph built: {self.graph_builder.num_suppliers} suppliers, "
              f"{self.graph_builder.num_warehouses} warehouses, "
              f"{self.graph_builder.num_retailers} retailers")
        
        # GNN Model
        self.gnn = SimplifiedRiskGNN(hidden_channels=64)
        print("✓ GNN model initialized")
        
        # RL Environment
        self.rl_env = SupplyChainEnv(self.config)
        self.rl_agent = SimpleRLOptimizer(self.rl_env)
        print("✓ RL agent initialized")
        
        print("\n✅ System ready!\n")
        
    def simulate_disruption(self, risk_node=None):
        """Simulate a disruption"""
        
        # Choose random supplier if none specified
        if risk_node is None:
            risk_node = np.random.randint(0, self.graph_builder.num_suppliers)
        
        print(f"\n🔔 DISRUPTION ALERT: Supplier {risk_node} experiencing issues!")
        
        # Set initial risks
        initial_risks = {'supplier': [risk_node]}
        
        # Propagate risk through GNN
        with torch.no_grad():
            risk_scores = self.gnn.predict_risk_propagation(
                self.node_features, self.adjacency, initial_risks
            )
        
        # Find affected nodes (risk > 0.5)
        affected_nodes = []
        for node_type, scores in risk_scores.items():
            if len(scores) > 0:
                high_risk = torch.where(scores > 0.5)[0].tolist()
                for idx in high_risk:
                    affected_nodes.append(f"{node_type}_{idx}")
        
        print(f"📊 Risk Propagation: {len(affected_nodes)} nodes affected")
        
        # Get RL recommendation
        observation = self.rl_env._get_observation()
        action = self.rl_agent.get_action(observation)
        
        action_desc = "Increase inventory" if action[1] > 0 else "Decrease inventory" if action[1] < 0 else "Maintain current levels"
        print(f"💡 Recommendation: {action_desc}")
        
        return {
            'risk_scores': risk_scores,
            'affected_nodes': affected_nodes,
            'recommended_action': action
        }
    
    def run_monitoring(self, days=10):
        """Run monitoring simulation"""
        print(f"\n📈 Running monitoring simulation for {days} days...")
        
        results = []
        for day in tqdm(range(days), desc="Monitoring"):
            # Simulate random disruption (30% chance)
            if np.random.random() < 0.3:
                disruption = np.random.randint(0, self.graph_builder.num_suppliers)
                assessment = self.simulate_disruption(disruption)
            else:
                assessment = self.simulate_disruption()
            
            # Calculate average risk
            all_scores = []
            for scores in assessment['risk_scores'].values():
                if len(scores) > 0:
                    all_scores.extend(scores.tolist())
            
            avg_risk = np.mean(all_scores) if all_scores else 0
            
            results.append({
                'day': day,
                'avg_risk': avg_risk,
                'num_affected': len(assessment['affected_nodes']),
                'action': assessment['recommended_action'][1] if len(assessment['recommended_action']) > 1 else 0
            })
            
            # Simulate one day
            self.rl_env.step(assessment['recommended_action'])
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv('monitoring_results.csv', index=False)
        print(f"\n✓ Results saved to monitoring_results.csv")
        
        return df
    
    def visualize(self):
        """Visualize the supply chain network"""
        print("\n🎨 Generating visualization...")
        
        G = nx.Graph()
        
        # Add nodes
        for i in range(self.graph_builder.num_suppliers):
            G.add_node(f"S{i}", type='supplier')
        for i in range(self.graph_builder.num_warehouses):
            G.add_node(f"W{i}", type='warehouse')
        for i in range(self.graph_builder.num_retailers):
            G.add_node(f"R{i}", type='retailer')
        
        # Add edges
        adj_sw = self.adjacency.get('supplier->warehouse')
        if adj_sw is not None:
            # Fix: properly iterate through torch.where results
            nonzero = torch.where(adj_sw > 0)
            for idx in range(len(nonzero[0])):
                w_idx = nonzero[0][idx].item()
                s_idx = nonzero[1][idx].item()
                G.add_edge(f"S{s_idx}", f"W{w_idx}")
        
        # Limit nodes for visualization (take first 50)
        if len(G.nodes()) > 50:
            nodes_list = list(G.nodes())[:50]
            G = G.subgraph(nodes_list)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create plot
        fig = go.Figure()
        
        # Edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(width=1, color='lightgray'),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Nodes by type
        colors = {'supplier': 'red', 'warehouse': 'blue', 'retailer': 'green'}
        for node_type, color in colors.items():
            nodes = [n for n in G.nodes() if n.startswith(node_type[0].upper())]
            if nodes:
                x = [pos[n][0] for n in nodes]
                y = [pos[n][1] for n in nodes]
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers+text',
                    marker=dict(size=12, color=color, opacity=0.7),
                    text=nodes,
                    textposition="bottom center",
                    name=node_type.title(),
                    hovertemplate='%{text}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Supply Chain Network Visualization',
            showlegend=True,
            width=1000,
            height=800,
            hovermode='closest'
        )
        
        fig.show()
        print("✓ Visualization complete!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function"""
    
    print("="*60)
    print("SUPPLY CHAIN AI SYSTEM - CPU OPTIMIZED")
    print("="*60)
    
    # Create default config if doesn't exist
    config = {
        'model_config': {
            'gnn': {'hidden_channels': 64},
            'llm': {'model_name': 'microsoft/phi-2'},
            'rl': {'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64, 'n_epochs': 10}
        },
        'supply_chain_config': {
            'num_suppliers': 20,
            'num_warehouses': 10,
            'num_retailers': 30,
            'num_products': 10,
            'simulation_days': 100
        }
    }
    
    import os
    if not os.path.exists('config.yaml'):
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
        print("✓ Created default config.yaml")
    
    # Initialize system
    try:
        system = SupplyChainAI('config.yaml')
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Quick Demo - Simulate a disruption")
        print("2. Run Monitoring (10 days)")
        print("3. Visualize Supply Chain")
        print("4. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Quick demo
            disruption = input("Enter supplier ID to disrupt (0-19, or press Enter for random): ").strip()
            if disruption:
                try:
                    disruption = int(disruption)
                except:
                    disruption = None
            else:
                disruption = None
            
            results = system.simulate_disruption(disruption)
            print(f"\n📊 Summary:")
            print(f"   - Affected nodes: {len(results['affected_nodes'])}")
            if results['affected_nodes']:
                print(f"   - Top affected: {results['affected_nodes'][:10]}")
            
        elif choice == '2':
            # Run monitoring
            days = input("Number of days to monitor (default 10): ").strip()
            days = int(days) if days else 10
            df = system.run_monitoring(days)
            print("\n📈 Monitoring Results Summary:")
            print(f"   - Average risk over period: {df['avg_risk'].mean():.4f}")
            print(f"   - Max risk: {df['avg_risk'].max():.4f}")
            print(f"   - Total disruptions: {df[df['num_affected'] > 0].shape[0]}")
            print("\nFirst few days:")
            print(df.head())
            
        elif choice == '3':
            # Visualize
            system.visualize()
            
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()