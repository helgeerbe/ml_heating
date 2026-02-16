"""
Model inspection and helper utilities for analysis.
Ported from notebooks/notebook_fix_helper.py.
"""

from typing import Any, Dict, Optional


def safe_get_regressor(model: Any) -> Optional[Any]:
    """
    Safely get the regressor from the model pipeline.

    Args:
        model: The model object (PhysicsCompliantWrapper, Pipeline,
               or Regressor)

    Returns:
        The underlying regressor object or None if not found.
    """
    try:
        # Handle PhysicsCompliantWrapper structure
        if hasattr(model, 'base_model'):
            base_model = model.base_model
            
            # Check if base model has steps
            if hasattr(base_model, 'steps'):
                steps = base_model.steps
                
                # Try different step names based on model type
                if 'learn' in steps:
                    return steps['learn']
                elif 'BaggingRegressor' in steps:
                    return steps['BaggingRegressor']
                elif len(steps) >= 2:
                    # Return the last step (usually the regressor)
                    step_names = list(steps.keys())
                    return steps[step_names[-1]]
            
            # Return base model directly
            return base_model
        
        # Try direct model steps
        if hasattr(model, 'steps'):
            steps = model.steps
            
            if 'learn' in steps:
                return steps['learn']
            elif 'BaggingRegressor' in steps:
                return steps['BaggingRegressor']
            elif len(steps) >= 2:
                # Return the last step
                step_names = list(steps.keys())
                return steps[step_names[-1]]
        
        # Return model directly as fallback
        return model
        
    except Exception as e:
        print(f"Error accessing regressor: {e}")
        return None


def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Get comprehensive model information for analysis.
    
    Args:
        model: The model object to inspect
        
    Returns:
        Dictionary containing model metadata and structure info.
    """
    info: Dict[str, Any] = {}
    
    try:
        info['model_type'] = type(model).__name__
        
        if hasattr(model, 'base_model'):
            info['base_model_type'] = type(model.base_model).__name__
            
            # Get base model steps
            if hasattr(model.base_model, 'steps'):
                info['base_steps'] = list(model.base_model.steps.keys())
        
        if hasattr(model, 'steps'):
            info['wrapper_steps'] = list(model.steps.keys())
        
        regressor = safe_get_regressor(model)
        if regressor:
            info['regressor_type'] = type(regressor).__name__
            info['regressor_available'] = True
            
            # Check for ensemble properties
            if hasattr(regressor, '__len__'):
                try:
                    info['ensemble_size'] = len(regressor)
                except Exception:
                    pass
                    
            if hasattr(regressor, 'n_models'):
                info['n_models'] = regressor.n_models
                
        else:
            info['regressor_available'] = False
            
    except Exception as e:
        info['error'] = str(e)
    
    return info
