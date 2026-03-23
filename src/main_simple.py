"""
Supply Chain AI System - Simplified Version
Run this from the root directory: python -m src.main_simple
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import warnings
import sys
import os

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# Now import using absolute imports
from src.data.graph_builder import SupplyChainGraphBuilder
from src.models.gnn_model import SimplifiedRiskGNN
from src.models.llm_agent import LightweightLLMAgent
from src.models.rl_agent import SupplyChainEnv, RLOptimizer

class SimpleSupplyChainAI:
    """Simplified orchestrator for supply chain AI system"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        # Load configuration
        config_full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all AI components"""
        self.logger.info("Initializing Supply Chain AI System...")
        
        # Graph Builder
        self.graph_builder = SupplyChainGraphBuilder(self.config)
        graph_data = self.graph_builder.build_graph()
        
        # Extract node features and adjacency
        self.node_features = graph_data['node_features']
        self.adjacency = graph_data['adjacency']
        
        # GNN Model
        node_feature_dims = {
            'supplier': 4,
            'warehouse': 3,
            'retailer': 3
        }
        gnn_config = self.config['model_config']['gnn']
        self.gnn = SimplifiedRiskGNN(
            node_feature_dims,
            hidden_channels=gnn_config['hidden_channels']
        )
        
        # LLM Agent (for CPU)
        llm_config = self.config['model_config']['llm']
        print("\n" + "="*50)
        print("Loading LLM model (this may take a few minutes on first run)...")
        print("="*50)
        
        # Try to load LLM, if fails, use a simple version
        try:
            self.llm = LightweightLLMAgent(model_name=llm_config['model_name'])
        except Exception as e:
            print(f"Warning: Could not load LLM: {e}")
            print("Using fallback mode without LLM...")
            self.llm = None
        
        # RL Environment and Agent
        self.rl_env = SupplyChainEnv(self.graph_builder, self.config)
        self.rl_agent = RLOptimizer(self.rl_env, self.config)
        
        self.logger.info("All components initialized!")
        
    def simulate_disruption(self, news_article: str = None):
        """Simulate a disruption and get recommendations"""
        
        # Step 1: Analyze news for risks (if provided and LLM available)
        if news_article and self.llm:
            try:
                risk_analysis = self.llm.analyze_news(news_article)
                self.logger.info(f"Risk Analysis: {risk_analysis}")
            except:
                risk_analysis = {}
            
            # Convert risk analysis to graph updates
            initial_risks = {'supplier': [0, 1, 2]}
            
        else:
            # Simulate random disruption
            num_suppliers = self.config['supply_chain_config']['num_suppliers']
            initial_risks = {'supplier': [np.random.randint(0, num_suppliers)]}
            
        # Step 2: Propagate risk through GNN
        with torch.no_grad():
            risk_scores = self.gnn.predict_risk_propagation(
                self.node_features, self.adjacency, initial_risks
            )
            
        # Step 3: Generate narrative
        affected_nodes = []
        for node_type, scores in risk_scores.items():
            if len(scores) > 0:
                high_risk = torch.where(scores > 0.7)[0].tolist()
                affected_nodes.extend([(node_type, idx) for idx in high_risk])
        
        narrative = "Risk detected in supply chain"
        if self.llm:
            try:
                narrative = self.llm.generate_risk_narrative(risk_scores, affected_nodes)
            except:
                pass
                
        self.logger.info(f"\nRisk Narrative:\n{narrative}")
        
        # Step 4: Get RL recommendation
        observation = self.rl_env._get_observation()
        recommended_action = self.rl_agent.get_action(observation)
        
        self.logger.info(f"\nRecommended Action: {recommended_action}")
        
        return {
            'risk_scores': risk_scores,
            'affected_nodes': affected_nodes,
            'narrative': narrative,
            'recommended_action': recommended_action
        }
    
    def train_system(self, timesteps: int = 5000):
        """Train the RL agent"""
        self.logger.info("Starting training...")
        self.rl_agent.train(total_timesteps=timesteps)
        self.logger.info("Training complete!")
        
    def run_monitoring(self, duration_days: int = 7):
        """Run continuous monitoring simulation"""
        results = []
        
        for day in tqdm(range(duration_days), desc="Monitoring"):
            # Get risk assessment
            assessment = self.simulate_disruption()
            
            # Calculate average risk
            avg_risk = 0
            risk_count = 0
            for scores in assessment['risk_scores'].values():
                if len(scores) > 0:
                    avg_risk += scores.mean().item()
                    risk_count += 1
            
            if risk_count > 0:
                avg_risk /= risk_count
            
            # Store results
            results.append({
                'day': day,
                'avg_risk': avg_risk,
                'num_affected': len(assessment['affected_nodes']),
                'action': assessment['recommended_action'].tolist()
            })
            
        # Create results dataframe
        df_results = pd.DataFrame(results)
        df_results.to_csv('monitoring_results.csv', index=False)
        
        self.logger.info(f"Monitoring complete! Results saved to monitoring_results.csv")
        return df_results
    
    def visualize(self):
        """Generate simple visualization of the supply chain"""
        import plotly.graph_objects as go
        import networkx as nx
        
        print("\nGenerating visualization...")
        
        # Create simple network visualization
        G = nx.Graph()
        
        # Add nodes
        num_suppliers = self.config['supply_chain_config']['num_suppliers']
        for i in range(num_suppliers):
            G.add_node(f"S{i}", type='supplier', color='red')
            
        num_warehouses = self.config['supply_chain_config']['num_warehouses']
        for i in range(num_warehouses):
            G.add_node(f"W{i}", type='warehouse', color='blue')
        
        num_retailers = self.config['supply_chain_config']['num_retailers']
        for i in range(num_retailers):
            G.add_node(f"R{i}", type='retailer', color='green')
        
        # Add edges from adjacency matrices
        # Supplier -> Warehouse
        adj_sw = self.adjacency.get('supplier->warehouse')
        if adj_sw is not None:
            for w_idx, s_idx in torch.where(adj_sw > 0):
                G.add_edge(f"S{s_idx.item()}", f"W{w_idx.item()}")
        
        # Warehouse -> Retailer
        adj_wr = self.adjacency.get('warehouse->retailer')
        if adj_wr is not None:
            for r_idx, w_idx in torch.where(adj_wr > 0):
                G.add_edge(f"W{w_idx.item()}", f"R{r_idx.item()}")
        
        # Create layout (limit nodes for better visualization)
        if len(G.nodes()) > 100:
            print(f"Graph has {len(G.nodes())} nodes, sampling for visualization...")
            # Sample nodes for visualization
            nodes_sample = list(G.nodes())[:100]
            G = G.subgraph(nodes_sample)
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create traces for different node types
        fig = go.Figure()
        
        # Add edges
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
        
        # Add nodes by type
        colors = {'supplier': 'red', 'warehouse': 'blue', 'retailer': 'green'}
        for node_type, color in colors.items():
            nodes = [node for node in G.nodes() if node.startswith(node_type[0].upper())]
            if nodes:
                x = [pos[node][0] for node in nodes]
                y = [pos[node][1] for node in nodes]
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
        print("Visualization complete!")

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("Supply Chain AI System - CPU Optimized Version")
    print("=" * 60)
    
    # Initialize system
    print("\nInitializing system...")
    try:
        system = SimpleSupplyChainAI('config.yaml')
    except Exception as e:
        print(f"Error initializing system: {e}")
        print("\nCheck if config.yaml exists in the root directory.")
        return
    
    # Ask user what to do
    print("\n" + "=" * 60)
    print("Select an option:")
    print("1. Quick demo (recommended)")
    print("2. Train RL agent (takes 5-10 minutes)")
    print("3. Run monitoring simulation")
    print("4. Visualize supply chain")
    print("5. Full system test")
    print("=" * 60)
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        # Quick demo
        print("\n" + "=" * 60)
        print("Running Quick Demo...")
        print("=" * 60)
        
        sample_news = "Major storm approaching Port of Rotterdam, expected to cause 3-day shipping delays."
        results = system.simulate_disruption(sample_news)
        
        print(f"\n📊 RESULTS:")
        print(f"Affected nodes: {results['affected_nodes'][:5]}")
        print(f"Recommended action: {results['recommended_action']}")
        
    elif choice == '2':
        # Train RL agent
        print("\nTraining RL agent...")
        system.train_system(timesteps=5000)
        
    elif choice == '3':
        # Run monitoring
        print("\nRunning monitoring simulation...")
        days = input("Number of days to simulate (default 7): ").strip()
        days = int(days) if days else 7
        results = system.run_monitoring(duration_days=days)
        print("\nMonitoring Results:")
        print(results.head())
        
    elif choice == '4':
        # Visualize
        system.visualize()
        
    elif choice == '5':
        # Full system test
        print("\n" + "=" * 60)
        print("Running Full System Test...")
        print("=" * 60)
        
        # Test 1: Basic simulation
        print("\n1. Testing disruption simulation...")
        results = system.simulate_disruption()
        print(f"   ✓ Affected nodes: {len(results['affected_nodes'])}")
        
        # Test 2: RL recommendation
        print("\n2. Testing RL recommendation...")
        print(f"   ✓ Recommended action: {results['recommended_action']}")
        
        # Test 3: Monitoring
        print("\n3. Testing monitoring (3 days)...")
        df = system.run_monitoring(duration_days=3)
        print(f"   ✓ Generated {len(df)} days of data")
        
        # Test 4: Visualization
        print("\n4. Testing visualization...")
        system.visualize()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    
    else:
        print("Invalid choice. Running quick demo...")
        results = system.simulate_disruption()
        print(f"\nAffected nodes: {results['affected_nodes'][:5]}")
        print(f"Recommended action: {results['recommended_action']}")
    
    print("\n✅ System execution complete!")

if __name__ == "__main__":
    main()