import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from _backtest import BackTest
import matplotlib.pyplot as plt


class TestBackTestVisualization(unittest.TestCase):
    """Visualization-specific tests for the BackTest class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for testing
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        assets = [f'Asset_{i}' for i in range(5)]
        
        # Factor data
        self.factor_df = pd.DataFrame(
            np.random.randn(50, 5), 
            index=dates, 
            columns=assets
        )
        
        # Price data
        self.price_df = pd.DataFrame(
            100 + np.random.randn(50, 5).cumsum(axis=0), 
            index=dates, 
            columns=assets
        )
        self.price_df['close'] = self.price_df.mean(axis=1)  # Add required 'close' column
        
        # Add market cap for market cap weighting tests
        self.price_df_with_mcap = self.price_df.copy()
        self.price_df_with_mcap['market_cap'] = np.random.uniform(1e8, 1e10, size=len(dates))
    
    def test_visualization_with_actual_plotting(self):
        """Test visualization methods with actual plotting (without mocking)."""
        bt = BackTest(
            factor_df=self.factor_df,
            price_df=self.price_df,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,  # Disable automatic plotting during init
            need_preprocess=False,
            cumprod=True
        )
        
        # Set up figures_path for actual file creation
        with tempfile.TemporaryDirectory() as temp_dir:
            bt.figures_path = temp_dir
            
            # Test actual plot_ret functionality
            try:
                bt.plot_ret()
                # Check if the file was created
                returns_file = os.path.join(temp_dir, 'returns.png')
                self.assertTrue(os.path.exists(returns_file), "returns.png should be created")
            except Exception as e:
                self.fail(f"plot_ret raised {type(e).__name__}: {e}")
            
            # Test actual plot_ic functionality
            try:
                bt.plot_ic()
                # Check if the file was created
                ic_file = os.path.join(temp_dir, 'ic.png')
                self.assertTrue(os.path.exists(ic_file), "ic.png should be created")
            except Exception as e:
                self.fail(f"plot_ic raised {type(e).__name__}: {e}")
            
            # Test actual plot_avg_group_ret functionality
            try:
                bt.plot_avg_group_ret()
                # Check if the file was created
                group_file = os.path.join(temp_dir, 'avg_group_returns.png')
                self.assertTrue(os.path.exists(group_file), "avg_group_returns.png should be created")
            except Exception as e:
                self.fail(f"plot_avg_group_ret raised {type(e).__name__}: {e}")
    
    def test_visualization_with_custom_figures_path(self):
        """Test visualization with custom figures path."""
        custom_path = os.path.join(tempfile.gettempdir(), 'custom_plots')
        
        # Ensure directory exists
        os.makedirs(custom_path, exist_ok=True)
        
        bt = BackTest(
            factor_df=self.factor_df,
            price_df=self.price_df,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False,
            cumprod=True,
            figures_path=custom_path  # Use custom path
        )
        
        # Test that the custom path is used
        self.assertEqual(bt.figures_path, custom_path)
        
        # Test actual plotting to custom path
        try:
            bt.plot_ret()
            returns_file = os.path.join(custom_path, 'returns.png')
            self.assertTrue(os.path.exists(returns_file), "returns.png should be created in custom path")
        except Exception as e:
            self.fail(f"plot_ret with custom path raised {type(e).__name__}: {e}")
        
        # Cleanup
        for file in os.listdir(custom_path):
            os.remove(os.path.join(custom_path, file))
        os.rmdir(custom_path)
    
    def test_visualization_with_auto_creation(self):
        """Test that figures_path is automatically created when None."""
        bt = BackTest(
            factor_df=self.factor_df,
            price_df=self.price_df,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False,
            cumprod=True,
            figures_path=None  # Should auto-create
        )
        
        # Check that figures_path was set to default
        expected_path = os.path.join(os.getcwd(), "figures_result")
        self.assertEqual(bt.figures_path, expected_path)
        
        # Test plotting to auto-created path (if directory exists)
        if os.path.exists(bt.figures_path):
            try:
                bt.plot_ret()
                returns_file = os.path.join(bt.figures_path, 'returns.png')
                self.assertTrue(os.path.exists(returns_file), "returns.png should be created in default path")
            except Exception as e:
                self.fail(f"plot_ret with auto path raised {type(e).__name__}: {e}")
    
    def test_visualization_methods_individual_call(self):
        """Test that each visualization method can be called individually."""
        bt = BackTest(
            factor_df=self.factor_df,
            price_df=self.price_df,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False,
            cumprod=True
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            bt.figures_path = temp_dir
            
            # Test individual method calls
            methods_to_test = ['plot_ret', 'plot_ic', 'plot_avg_group_ret']
            expected_files = ['returns.png', 'ic.png', 'avg_group_returns.png']
            
            for method_name, expected_file in zip(methods_to_test, expected_files):
                method = getattr(bt, method_name)
                
                # Call the method
                method()
                
                # Check if the file was created
                file_path = os.path.join(temp_dir, expected_file)
                self.assertTrue(
                    os.path.exists(file_path), 
                    f"{expected_file} should be created when calling {method_name}"
                )
    
    def test_visualization_with_real_data_plotting(self):
        """Test that visualization methods work with real data without errors."""
        # Create more realistic data to ensure plots work properly
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        assets = [f'STOCK_{i}' for i in range(10)]
        
        # Create more realistic factor data
        factor_data = pd.DataFrame(
            np.random.randn(100, 10), 
            index=dates, 
            columns=assets
        )
        
        # Create more realistic price data
        price_data = pd.DataFrame(
            100 * np.exp(np.random.randn(100, 10).cumsum(axis=0) * 0.01), 
            index=dates, 
            columns=assets
        )
        price_data['close'] = price_data.mean(axis=1)
        
        bt = BackTest(
            factor_df=factor_data,
            price_df=price_data,
            rebalance_period=20,
            n_groups=5,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False,
            cumprod=True
        )
        
        # Test plotting with more realistic data
        with tempfile.TemporaryDirectory() as temp_dir:
            bt.figures_path = temp_dir
            
            # These should not raise exceptions
            bt.plot_ret()
            bt.plot_ic()
            bt.plot_avg_group_ret()
            
            # Verify files exist
            for filename in ['returns.png', 'ic.png', 'avg_group_returns.png']:
                file_path = os.path.join(temp_dir, filename)
                self.assertTrue(os.path.exists(file_path), f"{filename} should exist")


if __name__ == '__main__':
    unittest.main()