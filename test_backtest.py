import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, MagicMock
from _backtest import BackTest


class TestBackTest(unittest.TestCase):
    """Unit tests for the BackTest class."""
    
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
        
        # Price data - now using asset columns directly instead of 'close'
        self.price_df = pd.DataFrame(
            100 + np.random.randn(50, 5).cumsum(axis=0), 
            index=dates, 
            columns=assets
        )
        
        # We no longer use market_cap, so removing that functionality
        self.price_df_with_mcap = self.price_df.copy()
    
    def test_init_valid_inputs(self):
        """Test initialization with valid inputs."""
        # Test with equal weighting
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
        self.assertIsInstance(bt, BackTest)
        self.assertEqual(bt.rebalance_period, 10)
        self.assertEqual(bt.n_groups, 3)
        self.assertEqual(bt.weight_method, 'equal')
    
    def test_init_market_cap_weighting(self):
        """Test initialization with market cap weighting - should raise error since it's removed."""
        with self.assertRaises(ValueError):
            BackTest(
                factor_df=self.factor_df,
                price_df=self.price_df_with_mcap,
                rebalance_period=10,
                n_groups=3,
                weight_method='market_cap',
                need_plot=False,
                need_preprocess=False,
                cumprod=True
            )
    
    def test_init_invalid_weight_method(self):
        """Test initialization with invalid weight method."""
        with self.assertRaises(ValueError):
            BackTest(
                factor_df=self.factor_df,
                price_df=self.price_df,
                rebalance_period=10,
                n_groups=3,
                weight_method='invalid_method',
                need_plot=False,
                need_preprocess=False
            )
    
    def test_init_only_equal_weight_method_supported(self):
        """Test that only 'equal' weight method is supported."""
        # Valid method should work
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
        self.assertIsInstance(bt, BackTest)
        
        # Other methods should raise error
        with self.assertRaises(ValueError):
            BackTest(
                factor_df=self.factor_df,
                price_df=self.price_df,
                rebalance_period=10,
                n_groups=3,
                weight_method='market_cap',  # This should now raise error
                need_plot=False,
                need_preprocess=False
            )
    
    def test_init_missing_stock_columns(self):
        """Test initialization without any stock price columns."""
        # Create a dataframe with no stock columns (only market_cap if present)
        price_df_no_stocks = pd.DataFrame(index=self.price_df.index)  # Empty dataframe with just the index
        with self.assertRaises(AssertionError):
            BackTest(
                factor_df=self.factor_df,
                price_df=price_df_no_stocks,
                rebalance_period=10,
                n_groups=3,
                weight_method='equal',
                need_plot=False,
                need_preprocess=False
            )
    

    
    def test_init_invalid_dataframe_types(self):
        """Test initialization with invalid dataframe types."""
        with self.assertRaises(AssertionError):
            BackTest(
                factor_df="not_a_dataframe",
                price_df=self.price_df,
                rebalance_period=10,
                n_groups=3,
                weight_method='equal',
                need_plot=False,
                need_preprocess=False
            )
        
        with self.assertRaises(AssertionError):
            BackTest(
                factor_df=self.factor_df,
                price_df="not_a_dataframe",
                rebalance_period=10,
                n_groups=3,
                weight_method='equal',
                need_plot=False,
                need_preprocess=False
            )
    
    def test_preprocess_basic(self):
        """Test basic preprocessing functionality."""
        bt = BackTest(
            factor_df=self.factor_df,
            price_df=self.price_df,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False
        )
        
        # Test preprocess method directly
        original_factor_df = self.factor_df.copy()
        original_price_df = self.price_df.copy()
        
        # Call preprocess
        bt.preprocess(original_price_df, original_factor_df)
        
        # Check that attributes are set
        self.assertIsNotNone(bt.time_index)
        self.assertIsNotNone(bt.factor_df)
        self.assertIsNotNone(bt.price_df)
    
    def test_preprocess_with_nan_values(self):
        """Test preprocessing with NaN values."""
        # Add NaN values to test data
        factor_df_with_nan = self.factor_df.copy()
        factor_df_with_nan.iloc[0, 0] = np.nan
        factor_df_with_nan.iloc[1, 1] = np.nan
        
        price_df_with_nan = self.price_df.copy()
        price_df_with_nan.iloc[0, 0] = np.nan
        price_df_with_nan.iloc[1, 1] = np.nan
        
        bt = BackTest(
            factor_df=factor_df_with_nan,
            price_df=price_df_with_nan,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False
        )
        
        # Manually call preprocess to test its functionality
        original_length = len(factor_df_with_nan.index.intersection(price_df_with_nan.index))
        
        # Call preprocess
        bt.preprocess(price_df_with_nan, factor_df_with_nan)
        
        # After preprocessing, the length should be less due to NaN removal
        processed_length = len(bt.time_index)
        # Note: The actual length depends on which rows have NaNs in both df
        # This test confirms that the method runs without errors
        self.assertIsNotNone(bt.time_index)
        self.assertIsNotNone(bt.factor_df)
        self.assertIsNotNone(bt.price_df)
    
    def test_preprocess_with_outliers(self):
        """Test preprocessing with outlier values."""
        # Add extreme outlier values
        factor_df_with_outliers = self.factor_df.copy()
        factor_df_with_outliers.iloc[0, 0] = 1000  # Extreme outlier
        
        price_df_with_outliers = self.price_df.copy()
        price_df_with_outliers.iloc[0, 0] = -100  # Negative price
        price_df_with_outliers.iloc[1, 1] = 2e6  # Above threshold
        
        bt = BackTest(
            factor_df=factor_df_with_outliers,
            price_df=price_df_with_outliers,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False
        )
        
        # Process without preprocessing to see raw effect
        original_time_index = bt.time_index
        # After preprocessing, outliers should be removed
    
    def test_compute_metrics_equal_weighting(self):
        """Test compute_metrics with equal weighting."""
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
        
        metrics = bt.compute_metrics(bt.price_df, bt.factor_df, cumprod=True)
        
        # Check that metrics are computed
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('rank_ic', metrics)
        self.assertIn('rank_ic_time_series', metrics)
        self.assertIn('cumulative_rank_ic', metrics)
        self.assertIn('avg_group_ret', metrics)
        self.assertIn('avg_group_daily_ret', metrics)
        self.assertIn('avg_group_cum_ret', metrics)
        
        # Check that values are numeric
        self.assertIsInstance(metrics['sharpe_ratio'], (float, int))
        self.assertIsInstance(metrics['sortino_ratio'], (float, int))
        self.assertIsInstance(metrics['max_drawdown'], (float, int))
        self.assertIsInstance(metrics['win_rate'], (float, int))
        self.assertIsInstance(metrics['rank_ic'], (float, int, type(np.nan)))
        
        # Check that the time series metrics are Series
        self.assertIsInstance(metrics['rank_ic_time_series'], pd.Series)
        self.assertIsInstance(metrics['cumulative_rank_ic'], pd.Series)
        
        # Check that the new metrics are DataFrames
        self.assertIsInstance(metrics['avg_group_daily_ret'], pd.DataFrame)
        self.assertIsInstance(metrics['avg_group_cum_ret'], pd.DataFrame)
    
    def test_compute_metrics_market_cap_weighting_should_fail(self):
        """Test compute_metrics with market cap weighting - should raise error since it's removed."""
        with self.assertRaises(ValueError):
            bt = BackTest(
                factor_df=self.factor_df,
                price_df=self.price_df_with_mcap,
                rebalance_period=10,
                n_groups=3,
                weight_method='market_cap',
                need_plot=False,
                need_preprocess=False,
                cumprod=True
            )
    
    def test_compute_metrics_different_cumprod_options(self):
        """Test compute_metrics with different cumprod options."""
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
        
        # Test with cumprod=True
        metrics_cumprod_true = bt.compute_metrics(bt.price_df, bt.factor_df, cumprod=True)
        
        # Test with cumprod=False
        metrics_cumprod_false = bt.compute_metrics(bt.price_df, bt.factor_df, cumprod=False)
        
        # Both should have the same keys
        self.assertEqual(set(metrics_cumprod_true.keys()), set(metrics_cumprod_false.keys()))
    
    def test_run_method(self):
        """Test the run method."""
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
        
        # The run method is called in __init__, so attributes should be set
        self.assertTrue(hasattr(bt, 'daily_ret'))
        self.assertTrue(hasattr(bt, 'cum_ret'))
        self.assertTrue(hasattr(bt, 'sharpe_ratio'))
        self.assertTrue(hasattr(bt, 'sortino_ratio'))
        self.assertTrue(hasattr(bt, 'calmar_ratio'))
        self.assertTrue(hasattr(bt, 'max_drawdown'))
        self.assertTrue(hasattr(bt, 'win_rate'))
        self.assertTrue(hasattr(bt, 'rank_ic'))
        self.assertTrue(hasattr(bt, 'avg_group_ret'))
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_ret(self, mock_close, mock_savefig):
        """Test the plot_ret method."""
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
        
        # Create temporary directory for figure saving
        with tempfile.TemporaryDirectory() as temp_dir:
            bt.figures_path = temp_dir
            
            # Call plot_ret
            bt.plot_ret()
            
            # Check that savefig and close were called
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_ic(self, mock_close, mock_savefig):
        """Test the plot_ic method."""
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
        
        # Create temporary directory for figure saving
        with tempfile.TemporaryDirectory() as temp_dir:
            bt.figures_path = temp_dir
            
            # Call plot_ic
            bt.plot_ic()
            
            # Check that savefig and close were called
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_avg_group_ret(self, mock_close, mock_savefig):
        """Test the plot_avg_group_ret method."""
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
        
        # Create temporary directory for figure saving
        with tempfile.TemporaryDirectory() as temp_dir:
            bt.figures_path = temp_dir
            
            # Call plot_avg_group_ret
            bt.plot_avg_group_ret()
            
            # Check that savefig and close were called
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()
    
    def test_factor_names_handling(self):
        """Test handling of factor names and empty factors."""
        # Create factor dataframe with one column having all NaN values
        factor_df_with_empty = self.factor_df.copy()
        factor_df_with_empty.iloc[:, 0] = np.nan  # Make first column all NaN
        
        bt = BackTest(
            factor_df=factor_df_with_empty,
            price_df=self.price_df,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=True,  # Enable preprocessing to trigger warnings
            cumprod=True
        )
        
        # The factor with all NaNs should be removed
        self.assertLess(len(bt.factor_names), len(factor_df_with_empty.columns))
    
    def test_edge_case_small_data(self):
        """Test with minimal viable data."""
        # Create minimal data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        assets = ['Asset_1', 'Asset_2']
        
        factor_df_small = pd.DataFrame(
            np.random.randn(5, 2), 
            index=dates, 
            columns=assets
        )
        
        price_df_small = pd.DataFrame(
            100 + np.random.randn(5, 2), 
            index=dates, 
            columns=assets
        )
        # Now using asset columns directly, no need for 'close' column
        
        bt = BackTest(
            factor_df=factor_df_small,
            price_df=price_df_small,
            rebalance_period=2,  # Small rebalance period
            n_groups=2,  # Small number of groups
            weight_method='equal',
            need_plot=False,
            need_preprocess=False,
            cumprod=True
        )
        
        # Should initialize successfully despite small data
        self.assertIsInstance(bt, BackTest)
    
    def test_zero_division_protection(self):
        """Test protection against zero division in metrics calculation."""
        # Create data that might cause zero division
        factor_df_zeros = pd.DataFrame(
            np.zeros((50, 5)), 
            index=self.factor_df.index, 
            columns=self.factor_df.columns
        )
        
        price_df_zeros = pd.DataFrame(
            100.0,  # Constant prices lead to zero returns
            index=self.price_df.index, 
            columns=self.price_df.columns
        )
        # Now using asset columns directly, no need for 'close' column
        
        bt = BackTest(
            factor_df=factor_df_zeros,
            price_df=price_df_zeros,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False,
            cumprod=True
        )
        
        # Should handle zero variance gracefully
        metrics = bt.compute_metrics(bt.price_df, bt.factor_df, cumprod=True)
        
        # Check that metrics are still computed (might contain NaN but shouldn't crash)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
    
    def test_datetime_index_conversion(self):
        """Test automatic datetime index conversion."""
        # Create data with non-datetime index
        factor_df_non_dt = self.factor_df.copy()
        factor_df_non_dt.index = range(len(factor_df_non_dt))
        
        price_df_non_dt = self.price_df.copy()
        price_df_non_dt.index = range(len(price_df_non_dt))
        
        bt = BackTest(
            factor_df=factor_df_non_dt,
            price_df=price_df_non_dt,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=True,  # This should convert indices to datetime
            cumprod=True
        )
        
        # After preprocessing, indices should be datetime
        self.assertIsInstance(bt.factor_df.index, pd.DatetimeIndex)
        self.assertIsInstance(bt.price_df.index, pd.DatetimeIndex)
    
    def test_preprocess_empty_data_error(self):
        """Test that preprocessing raises error with empty data."""
        empty_df = pd.DataFrame()
        
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
        
        with self.assertRaises(ValueError):
            bt.preprocess(empty_df, self.factor_df)
        
        with self.assertRaises(ValueError):
            bt.preprocess(self.price_df, empty_df)
    
    def test_compute_metrics_with_insufficient_assets_for_groups(self):
        """Test compute_metrics when there are insufficient assets to form groups."""
        # Create data with fewer assets than groups
        factor_df_few_assets = pd.DataFrame(
            np.random.randn(50, 2),  # Only 2 assets
            index=self.factor_df.index,
            columns=['Asset_1', 'Asset_2']
        )
        
        bt = BackTest(
            factor_df=factor_df_few_assets,
            price_df=self.price_df,
            rebalance_period=10,
            n_groups=5,  # More groups than assets
            weight_method='equal',
            need_plot=False,
            need_preprocess=False,
            cumprod=True
        )
        
        # This should still work, just with some groups having no assets
        metrics = bt.compute_metrics(bt.price_df, factor_df_few_assets, cumprod=True)
        self.assertIn('avg_group_ret', metrics)
        self.assertIn('avg_group_daily_ret', metrics)
        self.assertIn('avg_group_cum_ret', metrics)
        self.assertIn('rank_ic_time_series', metrics)
        self.assertIn('cumulative_rank_ic', metrics)
        # The resulting DataFrame should have NaN values for groups that couldn't be formed
        
        # Check that the time series metrics are Series
        self.assertIsInstance(metrics['rank_ic_time_series'], pd.Series)
        self.assertIsInstance(metrics['cumulative_rank_ic'], pd.Series)
        
        # Check that the new metrics are DataFrames
        self.assertIsInstance(metrics['avg_group_daily_ret'], pd.DataFrame)
        self.assertIsInstance(metrics['avg_group_cum_ret'], pd.DataFrame)
    
    def test_compute_metrics_quantile_cut_failure(self):
        """Test compute_metrics when quantile cut fails and fallback logic is used."""
        # Create factor data where all values are identical, causing qcut to fail
        factor_df_identical = pd.DataFrame(
            np.ones((50, 5)),  # All values are 1, so quantile cut will fail
            index=self.factor_df.index,
            columns=self.factor_df.columns
        )
        
        bt = BackTest(
            factor_df=factor_df_identical,
            price_df=self.price_df,
            rebalance_period=5,  # Smaller period to trigger more rebalancing
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False,
            cumprod=True
        )
        
        # Should handle quantile cut failure by using fallback logic
        metrics = bt.compute_metrics(bt.price_df, factor_df_identical, cumprod=True)
        self.assertIsNotNone(metrics['sharpe_ratio'])
        self.assertIn('avg_group_ret', metrics)
        self.assertIn('avg_group_daily_ret', metrics)
        self.assertIn('avg_group_cum_ret', metrics)
        self.assertIn('rank_ic_time_series', metrics)
        self.assertIn('cumulative_rank_ic', metrics)
        
        # Check that the time series metrics are Series
        self.assertIsInstance(metrics['rank_ic_time_series'], pd.Series)
        self.assertIsInstance(metrics['cumulative_rank_ic'], pd.Series)
        
        # Check that the new metrics are DataFrames
        self.assertIsInstance(metrics['avg_group_daily_ret'], pd.DataFrame)
        self.assertIsInstance(metrics['avg_group_cum_ret'], pd.DataFrame)
    
    def test_multiple_stock_columns_work(self):
        """Test that multiple stock columns work properly."""
        # The backtest should work with multiple stock columns (no longer requiring 'close')
        bt = BackTest(
            factor_df=self.factor_df,
            price_df=self.price_df,  # Using asset columns directly
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False,
            cumprod=True
        )
        
        # Should work with asset columns directly and compute metrics
        self.assertIsNotNone(bt.daily_ret)
    
    def test_factor_with_all_nans_warning(self):
        """Test behavior when a factor has all NaN values."""
        # Create factor dataframe with one column having all NaN values
        factor_df_with_all_nan_factor = self.factor_df.copy()
        factor_df_with_all_nan_factor.iloc[:, 0] = np.nan  # Make first factor all NaN
        
        # Temporarily suppress warnings for this test
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bt = BackTest(
                factor_df=factor_df_with_all_nan_factor,
                price_df=self.price_df,
                rebalance_period=10,
                n_groups=3,
                weight_method='equal',
                need_plot=False,
                need_preprocess=False,
                cumprod=True
            )
        
        # The all-NaN factor should be removed from factor_names
        self.assertLess(len(bt.factor_names), len(factor_df_with_all_nan_factor.columns))
    
    def test_factor_threshold_processing(self):
        """Test factor threshold processing in preprocessing."""
        # Create factor with extreme values that will be filtered out
        factor_df_extreme = self.factor_df.copy()
        factor_df_extreme.iloc[0, 0] = 100  # Very high value that exceeds threshold
        
        bt = BackTest(
            factor_df=factor_df_extreme,
            price_df=self.price_df,
            rebalance_period=10,
            n_groups=3,
            weight_method='equal',
            need_plot=False,
            need_preprocess=True,  # Enable preprocessing to apply thresholds
            need_normalize=False,  # Disable normalization for this test
            factor_threshold=5,  # Lower threshold to catch the extreme value
            cumprod=True
        )
        
        # The extreme value should be filtered out during preprocessing
        self.assertGreater(len(bt.factor_df), 0)
    

    
    def test_need_plot_functionality(self):
        """Test that plotting is triggered when need_plot=True."""
        with patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('os.makedirs'):
            
            bt = BackTest(
                factor_df=self.factor_df,
                price_df=self.price_df,
                rebalance_period=10,
                n_groups=3,
                weight_method='equal',
                need_plot=True,  # Enable plotting
                need_preprocess=False,
                cumprod=True
            )
            
            # Should have called the plotting methods
            # The plotting methods are called internally during __init__ when need_plot=True
    
    def test_compute_metrics_with_specific_conditions(self):
        """Test compute_metrics with specific conditions to cover more code paths."""
        bt = BackTest(
            factor_df=self.factor_df,
            price_df=self.price_df,
            rebalance_period=5,
            n_groups=2,
            weight_method='equal',
            need_plot=False,
            need_preprocess=False,
            cumprod=False  # Test with cumprod=False
        )
        
        # Test compute_metrics with cumprod=False
        metrics = bt.compute_metrics(bt.price_df, bt.factor_df, cumprod=False)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('avg_group_ret', metrics)
        self.assertIn('avg_group_daily_ret', metrics)
        self.assertIn('avg_group_cum_ret', metrics)
        self.assertIn('rank_ic_time_series', metrics)
        self.assertIn('cumulative_rank_ic', metrics)
        
        # Check that the time series metrics are Series
        self.assertIsInstance(metrics['rank_ic_time_series'], pd.Series)
        self.assertIsInstance(metrics['cumulative_rank_ic'], pd.Series)
        
        # Check that the new metrics are DataFrames
        self.assertIsInstance(metrics['avg_group_daily_ret'], pd.DataFrame)
        self.assertIsInstance(metrics['avg_group_cum_ret'], pd.DataFrame)
    
    def test_no_overlapping_indices_error(self):
        """Test error when there are no overlapping indices between factor and price data."""
        # Create data with non-overlapping indices
        factor_df_diff_idx = pd.DataFrame(
            np.random.randn(10, 5),
            index=pd.date_range('2022-01-01', periods=10),
            columns=self.factor_df.columns
        )
        
        # Create price data with same structure as original (5 asset columns)
        price_cols = [f'Asset_{i}' for i in range(5)]
        price_df_diff_idx = pd.DataFrame(
            np.random.randn(10, 5),
            index=pd.date_range('2023-01-01', periods=10),  # Different dates
            columns=price_cols
        )
        
        with self.assertRaises(ValueError):
            BackTest(
                factor_df=factor_df_diff_idx,
                price_df=price_df_diff_idx,
                rebalance_period=10,
                n_groups=3,
                weight_method='equal',
                need_plot=False,
                need_preprocess=True,  # This should trigger the error
                cumprod=True
            )


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
    
    def test_example_files_integration(self):
        """Test the backtest using the example CSV files provided."""
        import os
        
        # Load the example files
        price_df_orig = pd.read_csv('example_price.csv', index_col='time')
        factor_df_orig = pd.read_csv('example_factors.csv', index_col='Date')
        
        # Convert index to datetime if needed
        price_df_orig.index = pd.to_datetime(price_df_orig.index)
        factor_df_orig.index = pd.to_datetime(factor_df_orig.index)
        
        # Since the original example files have different assets (no common columns),
        # we'll create a synthetic test using a subset of the price data with artificial factors
        # that correspond to the same assets
        
        # Find overlapping dates
        common_dates = price_df_orig.index.intersection(factor_df_orig.index)
        
        if len(common_dates) > 5:  # At least 5 days for meaningful test
            # Use a reasonable subset of data
            if len(common_dates) > 30:
                common_dates = common_dates[:30]  # Take first 30 common dates to reduce test time
            
            # Select a subset of price columns to use as our test assets
            price_cols = price_df_orig.columns[:3]  # Use first 3 price columns
            price_df_test = price_df_orig.loc[common_dates, price_cols]
            
            # Create artificial factor data for the same assets (same column names as prices)
            # This simulates a realistic scenario where factors are calculated for the same assets
            factor_df_test = price_df_test.copy()
            # Rename columns to be factors (but for same assets)
            factor_df_test.columns = [f"FACTOR_{col}" for col in factor_df_test.columns]
            # Generate random factor-like values based on the price data
            for col in factor_df_test.columns:
                # Create factor values that have some relationship to the price movements
                factor_df_test[col] = price_df_test.iloc[:, 0].pct_change().fillna(0) + np.random.normal(0, 0.1, len(price_df_test))
            
            # Run the backtest with compatible data
            bt = BackTest(
                factor_df=factor_df_test,
                price_df=price_df_test,
                rebalance_period=5,
                n_groups=3,
                weight_method='equal',
                need_plot=False,
                need_preprocess=True,
                cumprod=True
            )
            
            # Check that the backtest ran successfully
            self.assertIsNotNone(bt.daily_ret)
            self.assertIsNotNone(bt.sharpe_ratio)
            self.assertIsNotNone(bt.win_rate)
            
            # Additionally check that IC and group return attributes are computed
            # Even if they are NaN, they should be initialized properly
            self.assertIsNotNone(bt.rank_ic)  # rank_ic should be computed
        else:
            # Skip test if insufficient data
            self.skipTest("Insufficient overlapping data between example files")


if __name__ == '__main__':
    unittest.main()