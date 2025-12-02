"""
Unit tests for model validation functionality

Tests the physics model validation script with train/test split
"""

import unittest
import sys
import os
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import from validation directory  
    validation_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'validation')
    sys.path.insert(0, validation_dir)
    
    # Import the validation script to get the ModelValidation class
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_model_validation", 
        os.path.join(validation_dir, "test_model_validation.py")
    )
    validation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_module)
    ModelValidation = validation_module.ModelValidation
    
    # Import other required classes - use same import path as validation module
    RealisticPhysicsModel = validation_module.RealisticPhysicsModel
    MAE = validation_module.MAE
    RMSE = validation_module.RMSE
except ImportError as e:
    unittest.skip(f"Skipping model validation tests due to import error: {e}")


class TestModelValidation(unittest.TestCase):
    """Test model validation with train/test split"""
    
    def setUp(self):
        """Set up test environment"""
        self.validator = ModelValidation()
        
        # Create mock data for testing
        self.mock_data = self._create_mock_training_data()
    
    def _create_mock_training_data(self) -> pd.DataFrame:
        """Create realistic mock training data"""
        np.random.seed(42)  # Reproducible tests
        
        n_samples = 1000
        timestamps = pd.date_range('2025-11-01', periods=n_samples, freq='10T')
        
        # Generate realistic feature data
        data = []
        for i, ts in enumerate(timestamps):
            # Simulate realistic temperature patterns
            outlet_temp = 35.0 + np.random.normal(0, 5)
            indoor_temp = 21.0 + np.random.normal(0, 0.5)
            outdoor_temp = 10.0 + np.random.normal(0, 3)
            target_temp = 21.0 + np.random.choice([-0.5, 0, 0.5])
            
            # Calculate realistic temperature change
            temp_diff = target_temp - indoor_temp
            heating_effect = (outlet_temp - 20) * 0.001
            actual_delta = temp_diff * 0.1 + heating_effect + np.random.normal(0, 0.05)
            
            sample = {
                'outlet_temp': outlet_temp,
                'indoor_temp_lag_30m': indoor_temp,
                'target_temp': target_temp,
                'outdoor_temp': outdoor_temp,
                'dhw_heating': np.random.choice([0, 1], p=[0.8, 0.2]),
                'dhw_disinfection': np.random.choice([0, 1], p=[0.95, 0.05]),
                'dhw_boost_heater': np.random.choice([0, 1], p=[0.9, 0.1]),
                'defrosting': np.random.choice([0, 1], p=[0.9, 0.1]),
                'pv_now': max(0, np.random.normal(500, 300)),
                'fireplace_on': np.random.choice([0, 1], p=[0.9, 0.1]),
                'tv_on': np.random.choice([0, 1], p=[0.7, 0.3]),
                'temp_forecast_1h': outdoor_temp + np.random.normal(0, 1),
                'temp_forecast_2h': outdoor_temp + np.random.normal(0, 1),
                'temp_forecast_3h': outdoor_temp + np.random.normal(0, 1),
                'temp_forecast_4h': outdoor_temp + np.random.normal(0, 1),
                'pv_forecast_1h': max(0, np.random.normal(400, 200)),
                'pv_forecast_2h': max(0, np.random.normal(300, 200)),
                'pv_forecast_3h': max(0, np.random.normal(200, 150)),
                'pv_forecast_4h': max(0, np.random.normal(100, 100)),
                'actual_delta': actual_delta,
                'timestamp': ts
            }
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def test_data_splitting(self):
        """Test train/test data splitting"""
        train_data, test_data = self.validator.split_data(self.mock_data)
        
        # Check split ratios
        total_samples = len(self.mock_data)
        expected_train = int(total_samples * 2/3)
        
        self.assertEqual(len(train_data), expected_train)
        self.assertEqual(len(test_data), total_samples - expected_train)
        
        # Check no overlap between train and test
        train_timestamps = set(train_data['timestamp'])
        test_timestamps = set(test_data['timestamp'])
        self.assertEqual(len(train_timestamps & test_timestamps), 0)
        
        # Check chronological ordering
        self.assertTrue(train_data['timestamp'].max() <= test_data['timestamp'].min())
    
    def test_model_training(self):
        """Test model training functionality"""
        train_data, _ = self.validator.split_data(self.mock_data)
        
        # Train model
        model, mae, rmse = self.validator.train_model(train_data)
        
        # Verify model type and metrics
        self.assertIsInstance(model, RealisticPhysicsModel)
        self.assertIsInstance(mae, MAE)
        self.assertIsInstance(rmse, RMSE)
        
        # Check that metrics have reasonable values
        self.assertGreater(mae.get(), 0.0)
        self.assertLess(mae.get(), 1.0)  # Should be reasonable for temperature prediction
        self.assertGreater(rmse.get(), 0.0)
        self.assertLess(rmse.get(), 1.5)
    
    def test_model_testing(self):
        """Test model evaluation on test data"""
        train_data, test_data = self.validator.split_data(self.mock_data)
        
        # Train model
        model, _, _ = self.validator.train_model(train_data)
        
        # Test model
        test_mae, test_rmse, detailed_results = self.validator.test_model(model, test_data)
        
        # Verify test metrics
        self.assertGreater(test_mae, 0.0)
        self.assertGreater(test_rmse, 0.0)
        self.assertGreaterEqual(test_rmse, test_mae)  # RMSE should be >= MAE
        
        # Check detailed results structure
        required_keys = ['test_mae', 'test_rmse', 'predictions', 'actuals', 
                        'errors', 'mean_error', 'std_error', 'max_error', 
                        'min_error', 'samples_tested']
        for key in required_keys:
            self.assertIn(key, detailed_results)
        
        # Check result consistency
        self.assertEqual(detailed_results['test_mae'], test_mae)
        self.assertEqual(detailed_results['test_rmse'], test_rmse)
        self.assertEqual(detailed_results['samples_tested'], len(test_data))
    
    def test_physics_behavior_testing(self):
        """Test physics behavior validation"""
        train_data, _ = self.validator.split_data(self.mock_data)
        model, _, _ = self.validator.train_model(train_data)
        
        # Test physics behavior (should not raise exceptions)
        try:
            self.validator.test_physics_behavior(model)
            physics_test_passed = True
        except Exception as e:
            physics_test_passed = False
            print(f"Physics behavior test failed: {e}")
        
        self.assertTrue(physics_test_passed, "Physics behavior test should not raise exceptions")
    
    def test_model_saving(self):
        """Test model saving functionality"""
        train_data, _ = self.validator.split_data(self.mock_data)
        model, mae, rmse = self.validator.train_model(train_data)
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            original_file = self.validator.validation_model_file
            self.validator.validation_model_file = tmp_file.name
            
            try:
                # Test saving
                self.validator.save_validation_model(model, mae, rmse)
                
                # Verify file was created
                self.assertTrue(os.path.exists(tmp_file.name))
                
                # Verify file size (should be non-zero)
                self.assertGreater(os.path.getsize(tmp_file.name), 0)
                
            finally:
                # Cleanup
                self.validator.validation_model_file = original_file
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
    
    def test_full_validation_pipeline_mock(self):
        """Test complete validation pipeline with mocked InfluxDB"""
        # Create mock validator with mocked InfluxDB service
        with patch.object(ModelValidation, 'load_and_prepare_data') as mock_load_data:
            mock_load_data.return_value = self.mock_data
            
            # Create validator 
            validator = ModelValidation()
            
            # Run validation with minimal data
            success = validator.run_full_validation(lookback_hours=24)
            
            # Verify validation completed
            self.assertTrue(success, "Full validation pipeline should complete successfully")
            
            # Verify data loading was called
            mock_load_data.assert_called_once_with(24)
    
    def test_train_test_data_quality(self):
        """Test that split data maintains quality"""
        train_data, test_data = self.validator.split_data(self.mock_data)
        
        # Check for required columns
        required_columns = ['outlet_temp', 'indoor_temp_lag_30m', 'target_temp', 
                          'outdoor_temp', 'actual_delta', 'timestamp']
        
        for col in required_columns:
            self.assertIn(col, train_data.columns)
            self.assertIn(col, test_data.columns)
        
        # Check for no NaN values in critical columns
        critical_columns = ['outlet_temp', 'indoor_temp_lag_30m', 'actual_delta']
        for col in critical_columns:
            self.assertFalse(train_data[col].isna().any(), f"No NaN values in {col}")
            self.assertFalse(test_data[col].isna().any(), f"No NaN values in {col}")
        
        # Check reasonable value ranges
        self.assertTrue((train_data['outlet_temp'] >= 10).all())
        self.assertTrue((train_data['outlet_temp'] <= 80).all())
        self.assertTrue((test_data['outlet_temp'] >= 10).all())
        self.assertTrue((test_data['outlet_temp'] <= 80).all())
        
        self.assertTrue((abs(train_data['actual_delta']) <= 5).all())
        self.assertTrue((abs(test_data['actual_delta']) <= 5).all())
    
    def test_performance_metrics_consistency(self):
        """Test that performance metrics are mathematically consistent"""
        train_data, test_data = self.validator.split_data(self.mock_data)
        model, _, _ = self.validator.train_model(train_data)
        test_mae, test_rmse, detailed_results = self.validator.test_model(model, test_data)
        
        # Manually calculate MAE and RMSE from predictions
        predictions = detailed_results['predictions']
        actuals = detailed_results['actuals']
        
        manual_mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
        manual_rmse = np.sqrt(np.mean([(a - p) ** 2 for a, p in zip(actuals, predictions)]))
        
        # Check consistency (allow small floating-point differences)
        self.assertAlmostEqual(test_mae, manual_mae, places=6)
        self.assertAlmostEqual(test_rmse, manual_rmse, places=6)
        self.assertAlmostEqual(detailed_results['mean_error'], manual_mae, places=6)


if __name__ == '__main__':
    unittest.main()
