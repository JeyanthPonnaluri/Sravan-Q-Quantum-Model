"""
Simple utilities for saving and loading model artifacts using joblib.
"""
import joblib
import os
from typing import Any


def save(obj: Any, path: str) -> None:
    """
    Save object to file using joblib.
    
    Args:
        obj: Object to save
        path: File path to save to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    joblib.dump(obj, path)
    print(f"Saved object to {path}")


def load(path: str) -> Any:
    """
    Load object from file using joblib.
    
    Args:
        path: File path to load from
        
    Returns:
        Loaded object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    obj = joblib.load(path)
    print(f"Loaded object from {path}")
    return obj


def save_models(models_dict: dict, base_dir: str = "models") -> None:
    """
    Save multiple models to a directory.
    
    Args:
        models_dict: Dictionary of {name: model} pairs
        base_dir: Base directory to save models
    """
    os.makedirs(base_dir, exist_ok=True)
    
    for name, model in models_dict.items():
        path = os.path.join(base_dir, f"{name}.pkl")
        save(model, path)


def load_models(model_names: list, base_dir: str = "models") -> dict:
    """
    Load multiple models from a directory.
    
    Args:
        model_names: List of model names (without .pkl extension)
        base_dir: Base directory containing models
        
    Returns:
        Dictionary of {name: model} pairs
    """
    models = {}
    
    for name in model_names:
        path = os.path.join(base_dir, f"{name}.pkl")
        models[name] = load(path)
    
    return models


if __name__ == "__main__":
    # Test save/load functionality
    import numpy as np
    
    # Test data
    test_data = {
        'array': np.random.rand(10, 5),
        'string': 'test_string',
        'number': 42
    }
    
    # Test save
    save(test_data, 'test_models/test_object.pkl')
    
    # Test load
    loaded_data = load('test_models/test_object.pkl')
    
    print("Save/load test completed successfully!")
    print(f"Original keys: {test_data.keys()}")
    print(f"Loaded keys: {loaded_data.keys()}")