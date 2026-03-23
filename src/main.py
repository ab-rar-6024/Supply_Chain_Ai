import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data.graph_builder import SupplyChainGraphBuilder
from models.gnn_model import RiskPropagationGNN, TemporalGNN
from models.llm_agent import LightweightLLMAgent
from models.rl_agent import SupplyChainEnv, RLOptimizer

class SupplyChainAI:
    """Main orchestrator for the supply chain AI system"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
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
        self.graph = self.graph_builder.build_graph()
        
        # GNN Model
        gnn_config = self.config['model_config']['gnn']
        self.gnn = RiskPropagationGNN(
            hidden_channels=gnn_config['hidden_channels'],
            num_layers=gnn_config['num_layers'],
            dropout=gnn_config['dropout']
        )
        
        # LLM Agent (for CPU)
        llm_config = self.config['model_config']['llm']
        self.llm = LightweightLLMAgent(model_name=llm_config['model_name'])
        
        # RL Environment and Agent
        self.rl_env = SupplyChainEnv(self.graph_builder, self.config)
        self.rl_agent = RLOptimizer(self.rl_env, self.config)
        
        self.logger.info("All components initialized!")
        
    def simulate_disruption(self, news_article: str = None):
        """Simulate a disruption and get recommendations"""
        
        # Step 1: Analyze news for risks (if provided)
        if news_article:
            risk_analysis = self.llm.analyze_news(news_article)
            self.logger.info(f"Risk Analysis: {risk_analysis}")
            
            # Convert risk analysis to graph updates
            initial_risks = {'supplier': [0, 1, 2]}  # Example risk nodes
            
        else:
            # Simulate random disruption
            initial_risks = {'supplier': [np.random.randint(0, 50)]}
            
        # Step 2: Propagate risk through GNN
        with torch.no_grad():
            risk_scores = self.gnn.predict_risk_propagation(self.graph, initial_risks)
            
        # Step 3: Generate narrative
        affected_nodes = []
        for node_type, scores in risk_scores.items():
            high_risk = torch.where(scores > 0.7)[0].tolist()
            affected_nodes.extend([(node_type, idx) for idx in high_risk])
            
        narrative = self.llm.generate_risk_narrative(risk_scores, affected_nodes)
        self.logger.info(f"Risk Narrative:\n{narrative}")
        
        # Step 4: Get RL recommendation
        observation = self.rl_env._get_observation()
        recommended_action = self.rl_agent.get_action(observation)
        
        self.logger.info(f"Recommended Action: {recommended_action}")
        
        return {
            'risk_scores': risk_scores,
            'affected_nodes': affected_nodes,
            'narrative': narrative,
            'recommended_action': recommended_action
        }
    
    def train_system(self, timesteps: int = 50000):
        """Train the RL agent"""
        self.logger.info("Starting training...")
        self.rl_agent.train(total_timesteps=timesteps)
        self.logger.info("Training complete!")
        
    def run_monitoring(self, duration_days: int = 30):
        """Run continuous monitoring simulation"""
        results = []
        
        for day in tqdm(range(duration_days), desc="Monitoring"):
            # Get risk assessment
            assessment = self.simulate_disruption()
            
            # Store results
            results.append({
                'day': day,
                'avg_risk': np.mean([s.mean().item() for s in assessment['risk_scores'].values()]),
                'num_affected': len(assessment['affected_nodes']),
                'action': assessment['recommended_action']
            })
            
        # Create results dataframe
        df_results = pd.DataFrame(results)
        df_results.to_csv('monitoring_results.csv', index=False)
        
        self.logger.info(f"Monitoring complete! Results saved to monitoring_results.csv")
        return df_results
    
    def visualize(self):
        """Generate visualizations of the supply chain"""
        import plotly.graph_objects as go
        import networkx as nx
        
        # Create simple network visualization
        G = nx.Graph()
        
        # Add nodes
        num_suppliers = self.config['supply_chain_config']['num_suppliers']
        for i in range(num_suppliers):
            G.add_node(f"S{i}", type='supplier')
            
        num_warehouses = self.config['supply_chain_config']['num_warehouses']
        for i in range(num_warehouses):
            G.add_node(f"W{i}", type='warehouse')
            
        # Add edges (simplified)
        edge_index = self.graph['supplier', 'supplies', 'warehouse'].edge_index
        for i in range(edge_index.size(1)):
            G.add_edge(f"S{edge_index[0,i]}", f"W{edge_index[1,i]}")
            
        # Create plot
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create traces for different node types
        supplier_nodes = [node for node in G.nodes() if node.startswith('S')]
        warehouse_nodes = [node for node in G.nodes() if node.startswith('W')]
        
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(width=1, color='gray'),
                showlegend=False
            ))
            
        # Add supplier nodes
        x = [pos[node][0] for node in supplier_nodes]
        y = [pos[node][1] for node in supplier_nodes]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(size=15, color='red'),
            text=supplier_nodes,
            name='Suppliers'
        ))
        
        # Add warehouse nodes
        x = [pos[node][0] for node in warehouse_nodes]
        y = [pos[node][1] for node in warehouse_nodes]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(size=15, color='blue'),
            text=warehouse_nodes,
            name='Warehouses'
        ))
        
        fig.update_layout(
            title='Supply Chain Network Visualization',
            showlegend=True,
            width=1000,
            height=800
        )
        
        fig.show()

def main():
    """Main execution function"""
    
    # Initialize system
    system = SupplyChainAI('config.yaml')
    
    # Option 1: Train the system
    print("Training RL agent...")
    system.train_system(timesteps=10000)  # Reduced for quick testing
    
    # Option 2: Run a disruption simulation
    print("\nSimulating disruption...")
    sample_news = "Major storm approaching Port of Rotterdam, expected to cause 3-day shipping delays."
    results = system.simulate_disruption(sample_news)
    
    print(f"\nResults:")
    print(f"Affected nodes: {results['affected_nodes'][:5]}")
    print(f"Recommended action: {results['recommended_action']}")
    
    # Option 3: Run monitoring
    print("\nRunning monitoring simulation...")
    monitoring_results = system.run_monitoring(duration_days=7)  # 7 days for quick test
    
    # Option 4: Visualize
    print("\nGenerating visualization...")
    system.visualize()
    
    print("\nSystem execution complete!")

if __name__ == "__main__":
    main()