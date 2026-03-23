import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, List, Tuple

class SupplyChainEnv(gym.Env):
    """Custom OpenAI Gym environment for supply chain simulation"""
    
    def __init__(self, graph_builder, config: Dict):
        super().__init__()
        
        self.graph_builder = graph_builder
        self.config = config
        self.current_day = 0
        
        # Action space: [supplier_switch, inventory_adjustment, route_reroute]
        self.action_space = spaces.Box(
            low=np.array([0, -100, 0], dtype=np.float32),
            high=np.array([1, 100, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space
        n_warehouses = config['supply_chain_config']['num_warehouses']
        n_retailers = config['supply_chain_config']['num_retailers']
        obs_dim = n_warehouses * 3 + n_retailers * 2 + 10
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        self.current_day = 0
        self.inventory = self._initialize_inventory()
        self.costs = []
        self.risk_history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in environment"""
        # Apply action
        supplier_switch, inventory_adj, route_change = action
        
        # Update state based on action
        self._apply_action(supplier_switch, inventory_adj, route_change)
        
        # Calculate costs
        holding_cost = self._calculate_holding_cost()
        transportation_cost = self._calculate_transportation_cost()
        penalty_cost = self._calculate_penalty_cost()
        
        total_cost = holding_cost + transportation_cost + penalty_cost
        self.costs.append(total_cost)
        
        # Calculate reward (negative cost + service level)
        reward = -total_cost + self._calculate_service_level()
        
        # Check if done
        self.current_day += 1
        done = self.current_day >= self.config['supply_chain_config']['simulation_days']
        
        # Get new observation
        obs = self._get_observation()
        
        return obs, reward, done, False, {'cost': total_cost}
    
    def _get_observation(self) -> np.ndarray:
        """Generate observation vector"""
        obs = []
        
        # Inventory levels
        for warehouse in self.inventory['warehouse']:
            obs.extend([warehouse['current'], warehouse['max'], warehouse['reorder_point']])
            
        for retailer in self.inventory['retailer']:
            obs.extend([retailer['current'], retailer['demand']])
            
        # Add padding if needed to match expected dimension
        expected_size = self.observation_space.shape[0]
        if len(obs) < expected_size:
            obs.extend([0] * (expected_size - len(obs)))
        elif len(obs) > expected_size:
            obs = obs[:expected_size]
            
        return np.array(obs, dtype=np.float32)
    
    def _initialize_inventory(self) -> Dict:
        """Initialize inventory levels"""
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
    
    def _apply_action(self, supplier_switch, inventory_adj, route_change):
        """Apply RL action to environment"""
        # Adjust inventory levels
        for warehouse in self.inventory['warehouse']:
            new_inv = warehouse['current'] + inventory_adj
            warehouse['current'] = np.clip(new_inv, 0, warehouse['max'])
            
        # Apply supplier switching (simplified)
        if supplier_switch > 0.5:
            # Switch to alternative supplier
            pass
            
        # Apply route changes
        if route_change > 0.5:
            # Reroute shipments
            pass
    
    def _calculate_holding_cost(self) -> float:
        """Calculate inventory holding cost"""
        total_inventory = sum(w['current'] for w in self.inventory['warehouse'])
        total_inventory += sum(r['current'] for r in self.inventory['retailer'])
        return total_inventory * 0.01  # $0.01 per unit holding cost
    
    def _calculate_transportation_cost(self) -> float:
        """Calculate transportation cost"""
        # Simplified: base cost per shipment
        return np.random.poisson(500)
    
    def _calculate_penalty_cost(self) -> float:
        """Calculate stockout penalty"""
        penalty = 0
        for retailer in self.inventory['retailer']:
            if retailer['current'] < retailer['demand']:
                penalty += (retailer['demand'] - retailer['current']) * 10
        return penalty
    
    def _calculate_service_level(self) -> float:
        """Calculate service level (fill rate)"""
        total_demand = sum(r['demand'] for r in self.inventory['retailer'])
        total_available = sum(r['current'] for r in self.inventory['retailer'])
        
        if total_demand > 0:
            return min(1.0, total_available / total_demand) * 100
        return 100

class RLOptimizer:
    """RL optimization for supply chain decisions"""
    
    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        
        # Use DummyVecEnv for single environment (no ray)
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Initialize PPO agent
        self.model = PPO(
            'MlpPolicy',
            self.vec_env,
            verbose=1,
            learning_rate=config['model_config']['rl']['learning_rate'],
            n_steps=config['model_config']['rl']['n_steps'],
            batch_size=config['model_config']['rl']['batch_size'],
            n_epochs=config['model_config']['rl']['n_epochs'],
            device='cpu'
        )
        
    def train(self, total_timesteps: int = 100000):
        """Train RL agent"""
        print(f"Training RL agent for {total_timesteps} timesteps...")
        try:
            self.model.learn(total_timesteps=total_timesteps)
            print("Training complete!")
        except Exception as e:
            print(f"Training error (this is normal if you skip training): {e}")
        
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Get action from trained policy"""
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            return action
        except:
            # Return default action if model not trained
            return np.array([0.5, 0, 0.5], dtype=np.float32)
    
    def save(self, path: str):
        """Save trained model"""
        self.model.save(path)
        
    def load(self, path: str):
        """Load trained model"""
        self.model = PPO.load(path, env=self.vec_env)