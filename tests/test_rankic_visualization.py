#!/usr/bin/env python3
"""
Test script to verify rankic visualization functionality
"""
import pandas as pd
import numpy as np
import os
import sys

# Import the BackTest class
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _backtest import BackTest

def create_test_data():
    """Create test data for backtesting"""
    # Create date range
    dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='B')
    
    # Create price data for 5 assets
    np.random.seed(42)
    price_data = np.random.randn(len(dates), 5).cumsum(axis=0) + 100
    price_df = pd.DataFrame(price_data, index=dates, columns=[f'Asset_{i+1}' for i in range(5)])
    
    # Create factor data (1 factor)
    factor_data = np.random.randn(len(dates), 1)
    factor_df = pd.DataFrame(factor_data, index=dates, columns=['Factor_1'])
    
    return price_df, factor_df

def test_rankic_visualization():
    """Test rankic visualization functionality"""
    print("Creating test data...")
    price_df, factor_df = create_test_data()
    
    print("Running backtest...")
    try:
        # Initialize backtest with plotting enabled
        bt = BackTest(
            factor_df=factor_df,
            price_df=price_df,
            rebalance_period=23,
            n_groups=3,
            weight_method='equal',
            need_plot=True,
            need_preprocess=True
        )
        
        print("Backtest completed successfully!")
        print(f"Overall Rank IC: {bt.rank_ic:.4f}")
        print(f"Number of Rank IC observations: {len(bt.rank_ic_time_series)}")
        
        # Test time series plot
        print("\nTesting time series plot...")
        bt.plot_ic(plot_type="time_series")
        
        # Test histogram plot
        print("Testing histogram plot...")
        bt.plot_ic(plot_type="histogram")
        
        # Check if figures were saved
        figures_path = bt.figures_path
        if os.path.exists(os.path.join(figures_path, "ic.png")):
            print(f"\n✓ Rank IC visualization saved successfully to: {figures_path}")
        else:
            print("\n✗ Rank IC visualization was not saved")
            
        return True
        
    except Exception as e:
        print(f"\n✗ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rankic_visualization()
    sys.exit(0 if success else 1)