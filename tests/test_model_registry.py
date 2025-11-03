"""
Tests for SURGE model registry.

Tests the extensibility and functionality of the model registry system.
"""

import pytest

from surge.models import (
    MODEL_REGISTRY,
    BaseModelAdapter,
    RandomForestModel,
    MLPModel,
)


class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_registry_list_models(self):
        """Test listing all registered models."""
        models = MODEL_REGISTRY.list_models()
        
        assert isinstance(models, dict), "list_models() should return a dict"
        assert len(models) > 0, "Registry should have at least one model"
        
        # Check for expected models (primary keys and aliases)
        keys = list(MODEL_REGISTRY.keys())
        assert 'sklearn.random_forest' in keys or 'random_forest' in keys, \
            f"Expected 'random_forest' or 'sklearn.random_forest' in {keys}"
        assert 'sklearn.mlp' in keys or 'mlp' in keys, \
            f"Expected 'mlp' or 'sklearn.mlp' in {keys}"
    
    def test_registry_create_model(self):
        """Test creating a model from registry using primary key and alias."""
        # Test using alias (common usage)
        model = MODEL_REGISTRY.create('random_forest', n_estimators=10)
        assert isinstance(model, BaseModelAdapter), "Should create BaseModelAdapter instance"
        assert isinstance(model, RandomForestModel), "Should create RandomForestModel instance"
        
        # Test using primary key
        model2 = MODEL_REGISTRY.create('sklearn.random_forest', n_estimators=20)
        assert isinstance(model2, RandomForestModel), "Should create RandomForestModel from primary key"
        
        # Test MLP model
        mlp_model = MODEL_REGISTRY.create('mlp', hidden_layer_sizes=(50,), max_iter=100)
        assert isinstance(mlp_model, MLPModel), "Should create MLPModel instance"
    
    def test_registry_keys(self):
        """Test getting all registry keys."""
        keys = list(MODEL_REGISTRY.keys())
        assert len(keys) > 0, "Registry should have keys"
        assert all(isinstance(k, str) for k in keys), "All keys should be strings"
        
        # Verify expected keys exist
        assert 'sklearn.random_forest' in keys or 'random_forest' in keys, \
            "Should have random_forest key or alias"
    
    def test_registry_get_metadata(self):
        """Test getting model metadata."""
        # Test using alias
        registered = MODEL_REGISTRY.get('random_forest')
        assert registered is not None, "Should return RegisteredModel"
        assert registered.adapter_cls is not None, "Should have adapter class"
        assert registered.adapter_cls == RandomForestModel, "Should be RandomForestModel"
        assert registered.key == 'sklearn.random_forest', "Primary key should be sklearn.random_forest"
        
        # Test using primary key
        registered2 = MODEL_REGISTRY.get('sklearn.random_forest')
        assert registered2 is not None, "Should return RegisteredModel from primary key"
        assert registered2.adapter_cls == RandomForestModel, "Should be RandomForestModel"
        
        # Test error handling for invalid key
        with pytest.raises(KeyError, match="not registered"):
            MODEL_REGISTRY.get('nonexistent_model')


class TestModelRegistryExtensibility:
    """Test that new models can be easily registered."""
    
    def test_register_custom_model(self):
        """Test registering a custom model adapter."""
        class CustomModelAdapter(BaseModelAdapter):
            name = "custom_test_model"
            backend = "test"
            
            def _build_model(self, **kwargs):
                # Dummy implementation
                return type('DummyModel', (), {'fit': lambda self, X, y: self, 'predict': lambda self, X: X[:, 0:1]})()
        
        # Register
        MODEL_REGISTRY.register(
            "custom_test",
            CustomModelAdapter,
            aliases=["test_model", "custom"]
        )
        
        # Verify primary key is registered
        assert "custom_test" in MODEL_REGISTRY.keys(), "Primary key should be registered"
        
        # Verify aliases are registered
        assert "test_model" in MODEL_REGISTRY.keys(), "Alias 'test_model' should be registered"
        assert "custom" in MODEL_REGISTRY.keys(), "Alias 'custom' should be registered"
        
        # Verify it appears in list_models
        models = MODEL_REGISTRY.list_models()
        assert "custom_test" in models, "Should appear in list_models()"
        assert models["custom_test"] == "CustomModelAdapter", "Should have correct class name"
        
        # Create instance using primary key
        model = MODEL_REGISTRY.create("custom_test", param1=1.0)
        assert isinstance(model, CustomModelAdapter), "Should create CustomModelAdapter instance"
        
        # Create instance using alias
        model2 = MODEL_REGISTRY.create("test_model", param1=2.0)
        assert isinstance(model2, CustomModelAdapter), "Should create from alias"
        
        # Verify metadata
        registered = MODEL_REGISTRY.get("custom_test")
        assert registered.adapter_cls == CustomModelAdapter, "Metadata should be correct"
        assert registered.key == "custom_test", "Should have correct primary key"
        
        # Note: Registry doesn't have unregister, so this test model persists
        # This is acceptable for testing and demonstrates extensibility

