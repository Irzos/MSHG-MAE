"""
Configuration utilities for masking strategies and experimental setups.

This module provides tools for loading, validating, and managing
configuration files for different masking strategies and experiments.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


class MaskingConfigValidator:
    """Validator for masking strategy configurations."""
    
    VALID_STRATEGIES = ['random', 'degree_aware', 'structure_aware', 'curriculum', 'adaptive']
    VALID_BANDIT_ALGORITHMS = ['ucb', 'thompson', 'epsilon_greedy']
    VALID_BLOCK_TYPES = ['functional_group', 'ring', 'chain', 'complex', 'aromatic']
    
    @classmethod
    def validate_masking_config(cls, config: Dict) -> Dict:
        """
        Validate masking configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration with defaults filled
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if 'masking' not in config:
            raise ConfigurationError("Missing 'masking' section in configuration")
        
        masking_config = config['masking']
        strategy = masking_config.get('strategy')
        
        if strategy not in cls.VALID_STRATEGIES:
            raise ConfigurationError(f"Invalid strategy '{strategy}'. Valid options: {cls.VALID_STRATEGIES}")
        
        # Validate mask ratio
        mask_ratio = masking_config.get('mask_ratio', 0.4)
        if not (0.0 < mask_ratio < 1.0):
            raise ConfigurationError(f"mask_ratio must be between 0 and 1, got {mask_ratio}")
        
        # Strategy-specific validation
        if strategy == 'curriculum':
            cls._validate_curriculum_config(masking_config)
        elif strategy == 'degree_aware':
            cls._validate_degree_aware_config(masking_config)
        elif strategy == 'structure_aware':
            cls._validate_structure_aware_config(masking_config)
        elif strategy == 'adaptive':
            cls._validate_adaptive_config(masking_config)
        
        return config
    
    @classmethod
    def _validate_curriculum_config(cls, masking_config: Dict):
        """Validate curriculum masking configuration."""
        curriculum_config = masking_config.get('curriculum', {})
        
        # Validate difficulty levels
        difficulty_levels = curriculum_config.get('difficulty_levels', 4)
        if not isinstance(difficulty_levels, int) or difficulty_levels < 1 or difficulty_levels > 10:
            raise ConfigurationError("difficulty_levels must be an integer between 1 and 10")
        
        # Validate bandit algorithm
        bandit_algorithm = curriculum_config.get('bandit_algorithm', 'ucb')
        if bandit_algorithm not in cls.VALID_BANDIT_ALGORITHMS:
            raise ConfigurationError(f"Invalid bandit_algorithm '{bandit_algorithm}'. Valid options: {cls.VALID_BANDIT_ALGORITHMS}")
        
        # Validate block types
        block_types = curriculum_config.get('block_types', ['functional_group', 'ring'])
        if not isinstance(block_types, list) or not block_types:
            raise ConfigurationError("block_types must be a non-empty list")
        
        for block_type in block_types:
            if block_type not in cls.VALID_BLOCK_TYPES:
                raise ConfigurationError(f"Invalid block_type '{block_type}'. Valid options: {cls.VALID_BLOCK_TYPES}")
        
        # Validate exploration factor
        exploration_factor = curriculum_config.get('exploration_factor', 1.5)
        if not isinstance(exploration_factor, (int, float)) or exploration_factor <= 0:
            raise ConfigurationError("exploration_factor must be a positive number")
    
    @classmethod
    def _validate_degree_aware_config(cls, masking_config: Dict):
        """Validate degree-aware masking configuration."""
        degree_config = masking_config.get('degree_aware', {})
        
        beta = degree_config.get('beta', 0.2)
        if not isinstance(beta, (int, float)) or beta < 0 or beta > 1:
            raise ConfigurationError("beta must be a number between 0 and 1")
    
    @classmethod
    def _validate_structure_aware_config(cls, masking_config: Dict):
        """Validate structure-aware masking configuration."""
        structure_config = masking_config.get('structure_aware', {})
        
        # Validate preservation probabilities
        for prob_key in ['ring_preservation_prob', 'fg_preservation_prob']:
            prob = structure_config.get(prob_key, 0.7)
            if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
                raise ConfigurationError(f"{prob_key} must be a number between 0 and 1")
    
    @classmethod
    def _validate_adaptive_config(cls, masking_config: Dict):
        """Validate adaptive masking configuration."""
        adaptive_config = masking_config.get('adaptive', {})
        
        strategies = adaptive_config.get('strategies', [])
        if not isinstance(strategies, list) or not strategies:
            raise ConfigurationError("adaptive.strategies must be a non-empty list")
        
        total_weight = 0
        for strategy in strategies:
            if not isinstance(strategy, dict):
                raise ConfigurationError("Each strategy in adaptive.strategies must be a dictionary")
            
            if 'type' not in strategy:
                raise ConfigurationError("Each strategy must have a 'type' field")
            
            if strategy['type'] not in cls.VALID_STRATEGIES:
                raise ConfigurationError(f"Invalid strategy type '{strategy['type']}'")
            
            weight = strategy.get('weight', 1.0)
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ConfigurationError("Strategy weight must be a non-negative number")
            
            total_weight += weight
        
        if total_weight == 0:
            raise ConfigurationError("Total weight of adaptive strategies must be positive")


class ConfigLoader:
    """Configuration loader with validation and defaults."""
    
    DEFAULT_CONFIG_PATHS = [
        'config/masking_strategies/',
        'config/',
        '.'
    ]
    
    @classmethod
    def load_config(cls, config_path: Union[str, Path], validate: bool = True) -> Dict:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            validate: Whether to validate the configuration
            
        Returns:
            Loaded and validated configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            # Try to find config in default paths
            for default_path in cls.DEFAULT_CONFIG_PATHS:
                full_path = Path(default_path) / config_path.name
                if full_path.exists():
                    config_path = full_path
                    break
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {config_path.suffix}")
            
            if validate:
                config = MaskingConfigValidator.validate_masking_config(config)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    @classmethod
    def load_ablation_configs(cls, ablation_config_path: Union[str, Path]) -> Dict[str, Dict]:
        """
        Load ablation study configurations.
        
        Args:
            ablation_config_path: Path to ablation configuration file
            
        Returns:
            Dictionary mapping experiment names to configurations
        """
        ablation_config = cls.load_config(ablation_config_path, validate=False)
        
        if 'ablation_configs' not in ablation_config:
            raise ConfigurationError("Missing 'ablation_configs' section")
        
        configs = {}
        base_config = ablation_config.get('base_config', {})
        
        for experiment_name, experiment_config in ablation_config['ablation_configs'].items():
            # Merge with base config
            merged_config = cls._deep_merge(base_config, experiment_config)
            
            # Validate the merged config
            try:
                merged_config = MaskingConfigValidator.validate_masking_config(merged_config)
                configs[experiment_name] = merged_config
            except ConfigurationError as e:
                logger.warning(f"Invalid configuration for experiment '{experiment_name}': {e}")
        
        logger.info(f"Loaded {len(configs)} valid ablation configurations")
        return configs
    
    @classmethod
    def _deep_merge(cls, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    @classmethod
    def save_config(cls, config: Dict, output_path: Union[str, Path]):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                with open(output_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            elif output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                raise ConfigurationError(f"Unsupported output format: {output_path.suffix}")
            
            logger.info(f"Saved configuration to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {output_path}: {e}")
            raise


class ExperimentConfigGenerator:
    """Generator for experiment configurations."""
    
    @classmethod
    def generate_grid_search_configs(cls, base_config: Dict, 
                                    parameter_grid: Dict[str, List]) -> Dict[str, Dict]:
        """
        Generate configurations for grid search.
        
        Args:
            base_config: Base configuration
            parameter_grid: Dictionary mapping parameter paths to value lists
            
        Returns:
            Dictionary mapping experiment names to configurations
        """
        import itertools
        
        # Get all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        configs = {}
        for i, combination in enumerate(itertools.product(*param_values)):
            config = deepcopy(base_config)
            experiment_name_parts = []
            
            for param_name, param_value in zip(param_names, combination):
                # Set parameter value in config
                cls._set_nested_parameter(config, param_name, param_value)
                # Add to experiment name
                experiment_name_parts.append(f"{param_name.replace('.', '_')}_{param_value}")
            
            experiment_name = "_".join(experiment_name_parts)
            configs[experiment_name] = config
        
        return configs
    
    @classmethod
    def _set_nested_parameter(cls, config: Dict, param_path: str, value: Any):
        """Set a nested parameter in configuration."""
        keys = param_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    @classmethod
    def generate_curriculum_ablation_configs(cls, base_config: Dict) -> Dict[str, Dict]:
        """Generate curriculum masking ablation configurations."""
        parameter_grid = {
            'masking.curriculum.difficulty_levels': [2, 3, 4, 5],
            'masking.curriculum.bandit_algorithm': ['ucb', 'thompson', 'epsilon_greedy'],
            'masking.curriculum.exploration_factor': [1.0, 1.5, 2.0],
            'masking.mask_ratio': [0.3, 0.4, 0.5]
        }
        
        return cls.generate_grid_search_configs(base_config, parameter_grid)


def load_masking_config(config_name_or_path: str, validate: bool = True) -> Dict:
    """
    Convenience function to load masking configuration.
    
    Args:
        config_name_or_path: Configuration name (e.g., 'curriculum_mask') or full path
        validate: Whether to validate the configuration
        
    Returns:
        Loaded configuration
    """
    if not config_name_or_path.endswith(('.yaml', '.yml', '.json')):
        # Try common naming patterns
        for pattern in [f"{config_name_or_path}_config.yaml", f"{config_name_or_path}.yaml"]:
            try:
                return ConfigLoader.load_config(pattern, validate)
            except FileNotFoundError:
                continue
        # If no pattern worked, use as-is
        config_name_or_path = f"{config_name_or_path}.yaml"
    
    return ConfigLoader.load_config(config_name_or_path, validate)


def create_experiment_configs(experiment_type: str = 'curriculum_ablation') -> Dict[str, Dict]:
    """
    Create experiment configurations for ablation studies.
    
    Args:
        experiment_type: Type of experiment ('curriculum_ablation', 'strategy_comparison', etc.)
        
    Returns:
        Dictionary mapping experiment names to configurations
    """
    base_config = load_masking_config('curriculum_mask_config')
    
    if experiment_type == 'curriculum_ablation':
        return ExperimentConfigGenerator.generate_curriculum_ablation_configs(base_config)
    elif experiment_type == 'strategy_comparison':
        # Load ablation configs
        return ConfigLoader.load_ablation_configs('masking_ablation_configs.yaml')
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


if __name__ == "__main__":
    # Example usage
    try:
        # Load and validate a curriculum masking config
        config = load_masking_config('curriculum_mask_config')
        print("✓ Curriculum masking config loaded successfully")
        print(f"Strategy: {config['masking']['strategy']}")
        print(f"Difficulty levels: {config['masking']['curriculum']['difficulty_levels']}")
        
        # Generate ablation study configs
        ablation_configs = create_experiment_configs('strategy_comparison')
        print(f"✓ Generated {len(ablation_configs)} ablation configurations")
        
        # Save example config
        ConfigLoader.save_config(config, 'example_output_config.yaml')
        print("✓ Example config saved")
        
    except Exception as e:
        print(f"✗ Error: {e}")