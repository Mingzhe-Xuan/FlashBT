import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

class Backtest:
    def __init__(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        fee: float,
        rebalance_period: int,
        n_groups: int,
        weight_method: str = "equal",
        market_cap_df: pd.DataFrame = None,
        need_preprocess: bool = True,
        need_normalize: bool = True,
        price_threshold: float = 1e6,
        factor_threshold: float = 10,
        need_plot: bool = True,
        metrics_path: str = None,
        figures_path: str = None,
        cumprod: bool = True,
        auto_run: bool = False,
        look_back: int = 69,
    ):
        r"""
        Back-test engine for factor-based strategies.

        Parameters
        ----------
        factor_df : pd.DataFrame
            Factor values for each asset at each time point. Note that the dataframe must have the same columns as price_df and has datetime as index.
        price_df : pd.DataFrame
            Close prices for each asset at each time point. Note that the dataframe must have the same columns as factor_df and has datetime as index.
        fee : float
            Transaction fee per trade (as a fraction of the trade amount).
            Fee is applied based on portfolio turnover at each rebalancing: fee_amount = portfolio_value * fee * turnover,
            where turnover = Σ|weight_change| / 2.
        rebalance_period : int
            Rebalancing frequency (number of periods between portfolio shifts). Day as the unit.
        n_groups : int
            Number of quantile groups into which assets are partitioned.
        weight_method : str
            Weighting scheme applied within each group. Options: "equal", "factor", "inv_vol", "mean_var", "market_cap". Default is "equal".
            - "equal": Equal weighting across all assets.
            - "factor": Weight assets proportionally to their factor values (linear factor weighting).
            - "inv_vol": Weight assets inversely proportional to their historical volatility.
            - "mean_var": Mean-variance optimization (Markowitz portfolio) maximizing Sharpe ratio.
            - "market_cap": Weight assets proportionally to their market capitalization (requires market_cap_df).
        market_cap_df : pd.DataFrame, optional
            Market capitalization data for each asset at each time point. Required if weight_method="market_cap".
            Note that dataframe must have same columns as price_df and has datetime as index.
        need_preprocess : bool
            Whether to preprocess data before back-testing. Default is True.
        need_normalize : bool
            Whether to normalize factor values. Default is True.
        price_threshold : float
            Upper bound of valid price values. Default is 1e6.
        factor_threshold : float
            The maximum times of standard deviation from mean is allowed for factor values. Default is 10.
        need_plot : bool
            Whether to plot results after back-testing. Default is True.
        metrics_path : str
            Path to save back-test metrics. Default is None.
        figures_path : str
            Path to save figures. Default is None.
        cumprod : bool
            Whether to compute cumulative product of daily returns. Default is True.
        auto_run : bool
            Whether automatically run the backtest as soon as the instance is created. Default is False.
        look_back : int
            Look-back period (in days) for calculating historical statistics used in weighting methods.
            Used in mean-variance optimization (60 days by default) and inverse volatility weighting (20 days).
            Default is 69.

        Attributes
        ----------
        daily_ret : pd.DataFrame
            Daily returns for each asset.
        cum_ret : pd.DataFrame
            Cumulative returns for each asset.
        portfolio_daily_ret : pd.Series
            Daily portfolio returns (with transaction fees applied based on turnover).
        portfolio_cum_ret : pd.Series
            Cumulative portfolio returns (with transaction fees applied based on turnover).
        sharpe_ratio : float
            Annualized Sharpe ratio (calculated from portfolio returns with turnover-based fees).
        sortino_ratio : float
            Annualized Sortino ratio (calculated from portfolio returns with turnover-based fees).
        calmar_ratio : float
            Annualized Calmar ratio (calculated from portfolio returns with turnover-based fees).
        max_drawdown : float
            Maximum drawdown experienced (calculated from portfolio returns with turnover-based fees).
        win_rate : float
            Fraction of positive-return periods (calculated from portfolio returns with turnover-based fees).
        ic : float
            Information coefficient (factor vs. forward return).
        rank_ic : float
            Rank information coefficient.
        avg_group_ret : pd.DataFrame
            Average return per group per period (with transaction fees applied based on turnover).
        avg_group_daily_ret : pd.DataFrame
            Daily returns per group (with transaction fees applied based on turnover).
        avg_group_cum_ret : pd.DataFrame
            Cumulative returns per group (with transaction fees applied based on turnover).

        Methods
        -------
        preprocess()
            Preprocess data for back-testing.
        run()
            Run back-test.
        compute_metrics()
            Compute back-test metrics.
        plot_ret()
            Plot cumulative and daily returns.
        plot_ic()
            Plot information-coefficient time series.
        plot_avg_group_ret()
            Plot average returns by group.
        """
        self.factor_df = factor_df
        self.price_df = price_df
        self.market_cap_df = market_cap_df
        self.fee = fee
        self.rebalance_period = rebalance_period
        self.n_groups = n_groups
        self.weight_method = weight_method
        self.need_normalize = need_normalize
        self.price_threshold = price_threshold
        self.factor_threshold = factor_threshold
        self.need_plot = need_plot
        self.metrics_path = metrics_path
        self.figures_path = figures_path
        self.cumprod = cumprod
        self.auto_run = auto_run
        self.look_back = look_back

        self.daily_ret = None
        self.cum_ret = None
        self.portfolio_daily_ret = None
        self.portfolio_cum_ret = None
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.calmar_ratio = None
        self.max_drawdown = None
        self.win_rate = None
        self.ic = None
        self.rank_ic = None
        self.avg_group_ret = None
        self.avg_group_daily_ret = None
        self.avg_group_cum_ret = None

        assert isinstance(
            self.factor_df, pd.DataFrame
        ), "factor_df must be a pandas DataFrame."
        assert isinstance(
            self.price_df, pd.DataFrame
        ), "price_df must be a pandas DataFrame."

        # Validate weight_method with only supported options
        # Allow for extensibility - only validate currently supported methods
        # Future weight methods can be added here as they are implemented
        supported_methods = [
            "equal",
            "factor",
            "inv_vol",
            "mean_var",
            "market_cap",
        ]  # Add new methods to this list as they are implemented
        if weight_method not in supported_methods:
            raise ValueError(
                f"Invalid weight_method '{weight_method}'. Supported methods: {supported_methods}"
            )

        # Validate market_cap_df if market_cap weighting is requested
        if weight_method == "market_cap" and market_cap_df is None:
            raise ValueError(
                "market_cap_df must be provided when weight_method='market_cap'"
            )

        assets_price = self.price_df.columns.tolist()
        assets_factor = self.factor_df.columns.tolist()
        assets_common = list(set(assets_price) & set(assets_factor))
        if self.market_cap_df is not None:
            assets_market_cap = self.market_cap_df.columns.tolist()
            assets_common = list(set(assets_common) & set(assets_market_cap))

        # Verify that we have at least one stock price column
        assert (
            len(assets_common) > 0
        ), "At least one common asset column between price_df, factor_df (and market_cap_df if it is not None) must be provided."

        self.price_df = self.price_df[assets_common]
        self.factor_df = self.factor_df[assets_common]
        if self.market_cap_df is not None:
            self.market_cap_df = self.market_cap_df[assets_common]

        if need_preprocess:
            self.preprocess(self.price_df, self.factor_df, self.market_cap_df)
        else:
            self.time_index = self.factor_df.index.intersection(self.price_df.index)
            self.price_df = self.price_df.loc[self.time_index].dropna(how="all")
            self.factor_df = self.factor_df.loc[self.time_index].dropna(how="all")
            if self.market_cap_df is not None:
                self.time_index = self.time_index.intersection(self.market_cap_df.index)
                self.price_df = self.price_df.loc[self.time_index]
                self.factor_df = self.factor_df.loc[self.time_index]
                self.market_cap_df = self.market_cap_df.loc[self.time_index]

        # Set stocks after preprocessing to ensure both dataframes have the same columns
        # self.stocks = self.factor_df.columns.tolist()
        # assert (
        #     self.stocks == self.price_df.columns.to_list()
        # ), "factor_df must have the same columns as price_df."        

        # Automatically run the backtest after initialization
        if self.auto_run:
            self.run()

    def preprocess(
        self,
        price_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        market_cap_df: pd.DataFrame = None,
    ):
        r"""
        Preprocess data for back-testing. The preprocessing steps include:
        1. Remove missing values (nans).
        2. Remove outliers (negative or zero prices, prices over price_threshold, factor values outside factor_threshold std from mean).
        3. Align factor and price dataframes by time index.
        4. Normalize factor values if it is required.

        Parameters
        ----------
        price_df : pd.DataFrame
            Close prices for each asset at each time point.
        factor_df : pd.DataFrame
            Factor values for each asset at each time point.
        market_cap_df : pd.DataFrame, optional
            Market capitalization data for each asset at each time point.

        Notes
        -----
        Transaction fees are applied based on portfolio turnover at each rebalancing:
        - Turnover = Σ|weight_change| / 2, where weight_change is the difference between
          current and previous portfolio weights
        - Fee amount = portfolio_value × fee × turnover
        - This properly reflects actual trading costs rather than assuming 100% turnover
        """
        if len(factor_df) == 0 or len(price_df) == 0:
            raise ValueError("No data provided.")

        factor_df = factor_df.dropna()
        price_df = price_df.dropna()
        
        # Process market_cap_df if provided
        if market_cap_df is not None:
            market_cap_df = market_cap_df.dropna()

        if len(factor_df) == 0 or len(price_df) == 0 or (market_cap_df is not None and len(market_cap_df) == 0):
            raise ValueError("No valid data after removing nans.")

        price_df = price_df[(price_df > 0) & (price_df < self.price_threshold)]
        
        # Apply market cap filtering (positive values only)
        if market_cap_df is not None:
            market_cap_df = market_cap_df[(market_cap_df > 0)]

        factor_mean = factor_df.mean()
        factor_std = factor_df.std()
        factor_df = factor_df[
            (factor_df > factor_mean - self.factor_threshold * factor_std)
            & (factor_df < factor_mean + self.factor_threshold * factor_std)
        ]

        if len(factor_df) == 0 or len(price_df) == 0 or (market_cap_df is not None and len(market_cap_df) == 0):
            raise ValueError("No valid data after removing nans and outliers.")

        if not isinstance(factor_df.index, pd.DatetimeIndex):
            factor_df.index = pd.to_datetime(factor_df.index)
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(price_df.index)
        if market_cap_df is not None and not isinstance(market_cap_df.index, pd.DatetimeIndex):
            market_cap_df.index = pd.to_datetime(market_cap_df.index)

        # Find intersection of all three dataframes' indices if market_cap_df is provided
        if market_cap_df is not None:
            time_index = factor_df.index.intersection(price_df.index).intersection(market_cap_df.index)
            if len(time_index) == 0:
                raise ValueError(
                    "No overlapping time index between valid factor, price, and market cap data."
                )
            factor_df = factor_df.loc[time_index]
            price_df = price_df.loc[time_index]
            market_cap_df = market_cap_df.loc[time_index]
        else:
            time_index = factor_df.index.intersection(price_df.index)
            if len(time_index) == 0:
                raise ValueError(
                    "No overlapping time index between valid factor and price data."
                )
            factor_df = factor_df.loc[time_index]
            price_df = price_df.loc[time_index]

        if self.need_normalize:
            factor_df = factor_df.apply(lambda x: (x - x.mean(axis=0)) / x.std(axis=0))

        self.time_index = time_index
        self.factor_df = factor_df
        self.price_df = price_df
        if market_cap_df is not None:
            self.market_cap_df = market_cap_df

    def compute_metrics(
        self,
        price_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        cumprod: bool = True,
    ) -> dict:
        r"""
        Compute back-test metrics. The metrics include:
        daily_ret : pd.DataFrame
            Daily returns for each asset.
        cum_ret : pd.DataFrame
            Cumulative returns for each asset.
        portfolio_daily_ret : pd.Series
            Daily portfolio returns (with transaction fees applied based on turnover).
        portfolio_cum_ret : pd.Series
            Cumulative portfolio returns (with transaction fees applied based on turnover).
        sharpe_ratio : float
            Annualized Sharpe ratio (calculated from portfolio returns with turnover-based fees).
        sortino_ratio : float
            Annualized Sortino ratio (calculated from portfolio returns with turnover-based fees).
        calmar_ratio : float
            Annualized Calmar ratio (calculated from portfolio returns with turnover-based fees).
        max_drawdown : float
            Maximum drawdown experienced (calculated from portfolio returns with turnover-based fees).
        win_rate : float
            Fraction of positive-return periods (calculated from portfolio returns with turnover-based fees).
        ic : float
            Information coefficient (factor vs. forward return).
        rank_ic : float
            Rank information coefficient.
        avg_group_ret : pd.DataFrame
            Average return per group per period (with transaction fees applied based on turnover).
        avg_group_daily_ret : pd.DataFrame
            Daily returns per group (with transaction fees applied based on turnover).
        avg_group_cum_ret : pd.DataFrame
            Cumulative returns per group (with transaction fees applied based on turnover).

        Parameters
        ----------
        price_df : pd.DataFrame
            Close prices for each asset at each time point.
        factor_df : pd.DataFrame
            Factor values for each asset at each time point.
        """
        cumprod = self.cumprod
        daily_ret_df = price_df.pct_change()
        if cumprod:
            cum_ret_df = (1 + daily_ret_df).cumprod() - 1
        else:
            cum_ret_df = daily_ret_df.cumsum()

        # Apply transaction fees based on turnover at rebalancing points
        # Track portfolio value and apply fees at rebalancing
        portfolio_value = 1.0  # Start with initial portfolio value of 1
        portfolio_daily_values = pd.Series(index=daily_ret_df.index, dtype=float)
        portfolio_daily_values.iloc[0] = portfolio_value

        # Track previous weights to calculate turnover
        prev_weights = None
        prev_assets = None

        rebalance_dates = factor_df.index[:: self.rebalance_period]

        # Precompute volatility for inv_vol method
        if self.weight_method == "inv_vol":
            # Calculate historical volatility for each asset
            # Use rolling window based on look_back parameter
            volatility_df = daily_ret_df.rolling(
                window=self.look_back, min_periods=max(10, self.look_back // 2)
            ).std()

        for i, date in enumerate(daily_ret_df.index):
            # Check if this is a rebalancing day
            is_rebalance = date in rebalance_dates

            if is_rebalance and i > 0:
                # Get assets that have valid returns on this date
                current_assets = daily_ret_df.columns[
                    daily_ret_df.loc[date].notna()
                ].tolist()

                if len(current_assets) > 0:
                    # Calculate weights based on weight_method
                    if self.weight_method == "equal":
                        # Equal weighting across available assets
                        current_weights = pd.Series(
                            1.0 / len(current_assets), index=current_assets
                        )

                    elif self.weight_method == "factor":
                        # Factor-based weighting: weight proportional to factor values
                        if date in factor_df.index:
                            factors_at_date = factor_df.loc[date]
                            # Get factors for current assets only
                            current_factors = factors_at_date[current_assets].dropna()
                            valid_assets = current_factors.index.tolist()

                            if len(valid_assets) > 0:
                                # Shift factors to be positive if needed
                                factor_values = current_factors.values
                                if np.any(factor_values < 0):
                                    factor_values = (
                                        factor_values - factor_values.min() + 1e-6
                                    )

                                # Normalize to sum to 1
                                current_weights = pd.Series(
                                    factor_values / factor_values.sum(),
                                    index=valid_assets,
                                )
                            else:
                                # Fallback to equal weighting
                                current_weights = pd.Series(
                                    1.0 / len(current_assets), index=current_assets
                                )
                        else:
                            # Fallback to equal weighting if factor data not available
                            current_weights = pd.Series(
                                1.0 / len(current_assets), index=current_assets
                            )

                    elif self.weight_method == "inv_vol":
                        # Inverse volatility weighting
                        if date in volatility_df.index:
                            vols_at_date = volatility_df.loc[date]
                            # Get volatilities for current assets only
                            current_vols = vols_at_date[current_assets].dropna()
                            valid_assets = current_vols.index.tolist()

                            if len(valid_assets) > 0:
                                # Calculate inverse volatilities
                                inv_vols = 1.0 / (
                                    current_vols.values + 1e-6
                                )  # Add small constant to avoid division by zero

                                # Normalize to sum to 1
                                current_weights = pd.Series(
                                    inv_vols / inv_vols.sum(), index=valid_assets
                                )
                            else:
                                # Fallback to equal weighting
                                current_weights = pd.Series(
                                    1.0 / len(current_assets), index=current_assets
                                )
                        else:
                            # Fallback to equal weighting if volatility data not available
                            current_weights = pd.Series(
                                1.0 / len(current_assets), index=current_assets
                            )

                    elif self.weight_method == "mean_var":
                        # Mean-variance optimization (Markowitz portfolio)
                        # Use historical returns to estimate mean and covariance
                        lookback_period = self.look_back

                        if i >= lookback_period:
                            # Get historical returns for lookback period
                            hist_returns = daily_ret_df.iloc[i - lookback_period : i]

                            # Filter to current assets
                            hist_returns = hist_returns[current_assets].dropna(
                                axis=1, how="all"
                            )
                            valid_assets = hist_returns.columns.tolist()

                            if len(valid_assets) > 1:
                                # Calculate expected returns (mean) and covariance matrix
                                mu = hist_returns.mean().values
                                cov_matrix = hist_returns.cov().values

                                # Add small regularization to covariance matrix for numerical stability
                                cov_matrix = (
                                    cov_matrix + np.eye(len(valid_assets)) * 1e-8
                                )

                                try:
                                    # Markowitz optimal weights: w* = Σ^-1 * μ / (1^T * Σ^-1 * μ)
                                    cov_inv = np.linalg.inv(cov_matrix)
                                    ones = np.ones(len(valid_assets))

                                    # Calculate numerator: Σ^-1 * μ
                                    numerator = cov_inv @ mu

                                    # Calculate denominator: 1^T * Σ^-1 * μ
                                    denominator = ones @ numerator

                                    # Calculate optimal weights
                                    if abs(denominator) > 1e-10:
                                        optimal_weights = numerator / denominator

                                        # Ensure weights are non-negative (long-only constraint)
                                        optimal_weights = np.maximum(optimal_weights, 0)

                                        # Normalize to sum to 1
                                        if optimal_weights.sum() > 0:
                                            optimal_weights = (
                                                optimal_weights / optimal_weights.sum()
                                            )
                                        else:
                                            # Fallback to equal weighting
                                            optimal_weights = np.ones(
                                                len(valid_assets)
                                            ) / len(valid_assets)
                                    else:
                                        # Fallback to equal weighting
                                        optimal_weights = np.ones(
                                            len(valid_assets)
                                        ) / len(valid_assets)

                                    current_weights = pd.Series(
                                        optimal_weights, index=valid_assets
                                    )
                                except (np.linalg.LinAlgError, ValueError):
                                    # Fallback to equal weighting if matrix inversion fails
                                    current_weights = pd.Series(
                                        1.0 / len(valid_assets), index=valid_assets
                                    )
                            else:
                                # Fallback to equal weighting
                                current_weights = pd.Series(
                                    1.0 / len(current_assets), index=current_assets
                                )
                        else:
                            # Not enough historical data, fallback to equal weighting
                            current_weights = pd.Series(
                                1.0 / len(current_assets), index=current_assets
                            )

                    elif self.weight_method == "market_cap":
                        # Market capitalization weighting
                        if (
                            self.market_cap_df is not None
                            and date in self.market_cap_df.index
                        ):
                            market_caps_at_date = self.market_cap_df.loc[date]
                            # Get market caps for current assets only
                            current_market_caps = market_caps_at_date[
                                current_assets
                            ].dropna()
                            valid_assets = current_market_caps.index.tolist()

                            if len(valid_assets) > 0:
                                # Calculate weights proportional to market cap
                                market_cap_values = current_market_caps.values

                                # Ensure all market caps are positive
                                market_cap_values = np.maximum(market_cap_values, 0)

                                # Normalize to sum to 1
                                if market_cap_values.sum() > 0:
                                    current_weights = pd.Series(
                                        market_cap_values / market_cap_values.sum(),
                                        index=valid_assets,
                                    )
                                else:
                                    # Fallback to equal weighting
                                    current_weights = pd.Series(
                                        1.0 / len(valid_assets), index=valid_assets
                                    )
                            else:
                                # Fallback to equal weighting
                                current_weights = pd.Series(
                                    1.0 / len(current_assets), index=current_assets
                                )
                        else:
                            # Fallback to equal weighting if market cap data not available
                            current_weights = pd.Series(
                                1.0 / len(current_assets), index=current_assets
                            )

                    else:
                        # Default to equal weighting for any unrecognized method
                        current_weights = pd.Series(
                            1.0 / len(current_assets), index=current_assets
                        )

                    # Calculate turnover if we have previous weights
                    if prev_weights is not None and prev_assets is not None:
                        # Calculate turnover: sum of absolute weight changes / 2
                        # Create aligned weight series
                        all_assets = list(set(prev_assets + current_assets))
                        prev_weights_aligned = pd.Series(0.0, index=all_assets)
                        current_weights_aligned = pd.Series(0.0, index=all_assets)

                        for asset in prev_assets:
                            if asset in prev_weights.index:
                                prev_weights_aligned[asset] = prev_weights[asset]

                        for asset in current_assets:
                            if asset in current_weights.index:
                                current_weights_aligned[asset] = current_weights[asset]

                        # Calculate turnover
                        turnover = (
                            current_weights_aligned - prev_weights_aligned
                        ).abs().sum() / 2.0

                        # Apply fee based on turnover
                        fee_amount = portfolio_value * self.fee * turnover
                        portfolio_value -= fee_amount

                    # Update previous weights and assets
                    prev_weights = current_weights
                    prev_assets = current_assets

            # Apply daily return using current weights
            if i > 0:
                # Calculate weighted return for this day
                if prev_weights is not None and len(prev_weights) > 0:
                    # Get returns for assets in current portfolio
                    returns_at_date = daily_ret_df.loc[date]
                    weighted_return = 0.0

                    for asset, weight in prev_weights.items():
                        if asset in returns_at_date.index and pd.notna(
                            returns_at_date[asset]
                        ):
                            weighted_return += weight * returns_at_date[asset]

                    portfolio_value *= 1 + weighted_return

            portfolio_daily_values.iloc[i] = portfolio_value

        # Calculate daily returns from portfolio values
        portfolio_daily_ret = portfolio_daily_values.pct_change().fillna(0)

        # Calculate cumulative returns from portfolio values
        if cumprod:
            portfolio_cum_ret = (
                portfolio_daily_values / portfolio_daily_values.iloc[0]
            ) - 1
        else:
            portfolio_cum_ret = portfolio_daily_values - portfolio_daily_values.iloc[0]

        # Set avg_daily_returns and avg_cum_ret to use fee-adjusted returns
        avg_daily_returns = portfolio_daily_ret
        avg_cum_ret = portfolio_cum_ret

        # Calculate Sharpe ratio with protection against division by zero
        daily_mean = avg_daily_returns.mean()
        daily_std = avg_daily_returns.std()
        if daily_std == 0 or pd.isna(daily_std) or pd.isna(daily_mean):
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252)

        # Calculate Sortino ratio with protection against division by zero
        negative_returns = avg_daily_returns[avg_daily_returns < 0]
        if len(negative_returns) > 0:
            negative_std = negative_returns.std()
            if negative_std == 0 or pd.isna(negative_std) or pd.isna(daily_mean):
                sortino_ratio = 0.0
            else:
                sortino_ratio = (daily_mean / negative_std) * np.sqrt(252)
        else:
            # No negative returns, use regular std as fallback
            sortino_ratio = sharpe_ratio

        # Calculate Calmar ratio with protection against division by zero
        daily_min = avg_daily_returns.min()
        if daily_min == 0 or pd.isna(daily_min) or pd.isna(daily_mean):
            calmar_ratio = 0.0
        else:
            calmar_ratio = (daily_mean / abs(daily_min)) * np.sqrt(252)

        # Calculate max drawdown from cumulative returns
        if cumprod:
            running_max = (1 + avg_cum_ret).cummax()
            drawdown = (1 + avg_cum_ret) / running_max - 1
        else:
            running_max = avg_cum_ret.expanding().max()
            drawdown = avg_cum_ret - running_max

        # Calculate max drawdown with NaN protection
        if drawdown.empty or pd.isna(drawdown.min()):
            max_drawdown = 0.0
        else:
            max_drawdown = drawdown.min()

        # Calculate win rate with NaN protection
        if avg_daily_returns.empty or pd.isna((avg_daily_returns > 0).mean()):
            win_rate = 0.0
        else:
            win_rate = (avg_daily_returns > 0).mean()

        # Calculate Rank IC for each time point for visualization
        # Use factors at time t to predict returns at time t+1 to avoid forward-looking bias
        rank_ic_series = []
        rank_ic_dates = []

        # Get the list of dates in order
        factor_dates = factor_df.index
        for i, date in enumerate(
            factor_dates[:-1]
        ):  # Exclude the last date since there's no future return
            if date in daily_ret_df.index:
                # Get factors at current date (time t)
                factors_at_date = factor_df.loc[date].dropna()

                # Get returns at the next date (time t+1) - this avoids forward-looking bias
                next_date_idx = i + 1
                if (
                    next_date_idx < len(factor_dates)
                    and factor_dates[next_date_idx] in daily_ret_df.index
                ):
                    returns_at_next_date = daily_ret_df.loc[
                        factor_dates[next_date_idx]
                    ].dropna()

                    # Find common assets
                    common_assets = factors_at_date.index.intersection(
                        returns_at_next_date.index
                    )
                    if len(common_assets) > 1:  # Need at least 2 points for correlation
                        aligned_factors = factors_at_date[common_assets]
                        aligned_returns = returns_at_next_date[common_assets]

                        # Calculate Rank IC for this date using current factors to predict next returns
                        rank_ic_val = aligned_factors.rank().corr(
                            aligned_returns, method="spearman"
                        )

                        if not pd.isna(rank_ic_val):
                            rank_ic_series.append(rank_ic_val)
                            rank_ic_dates.append(date)

        # Also calculate overall Rank IC across all time periods and assets
        # Use factors at time t to predict returns at time t+1 to avoid forward-looking bias
        # Shift returns forward by one period to align factors with future returns
        factor_values = factor_df.stack()
        shifted_return_df = daily_ret_df.shift(
            -1
        )  # Shift returns back by 1 so factor t predicts return t+1
        return_values = shifted_return_df.reindex(factor_df.index).stack()

        # Only keep pairs where both factor and return exist
        common_idx = factor_values.index.intersection(return_values.index)
        if len(common_idx) > 1:  # Need at least 2 points for correlation
            aligned_factors = factor_values[common_idx]
            aligned_returns = return_values[common_idx]
            overall_rank_ic = aligned_factors.rank().corr(
                aligned_returns, method="spearman"
            )
            if pd.isna(overall_rank_ic):
                overall_rank_ic = 0.0
        else:
            overall_rank_ic = 0.0

        # Create time series for Rank IC
        rank_ic_time_series = (
            pd.Series(data=rank_ic_series, index=rank_ic_dates, name="Rank_IC")
            if rank_ic_dates
            else pd.Series(dtype=float)
        )

        # Calculate cumulative Rank IC
        cumulative_rank_ic = (
            rank_ic_time_series.cumsum()
            if len(rank_ic_time_series) > 0
            else pd.Series(dtype=float)
        )

        # Group by factor value at each rebalance moment and calculate return separately
        # Create groups based on factor quantiles at each rebalance period
        rebalance_dates = factor_df.index[:: self.rebalance_period]
        avg_group_ret_by_period = pd.DataFrame(
            index=rebalance_dates, columns=range(self.n_groups)
        )

        # Create a DataFrame to store daily group returns for visualization
        all_group_daily_returns = {}
        all_group_portfolio_values = {}  # Track portfolio values for each group
        all_group_prev_weights = {}  # Track previous weights for turnover calculation
        all_group_prev_assets = {}  # Track previous assets for turnover calculation

        for date in rebalance_dates:
            # Get factor values at this rebalance date
            current_factors = factor_df.loc[date].dropna()

            # Create quantile groups based on factor values
            if (
                len(current_factors) >= self.n_groups
            ):  # Need enough assets to form groups
                try:
                    # Assign assets to groups based on factor quantiles
                    groups = pd.qcut(
                        current_factors,
                        q=self.n_groups,
                        labels=False,
                        duplicates="drop",
                    )

                    # Calculate returns for each group in the subsequent period
                    # Calculate returns for the next rebalance_period days
                    date_idx = factor_df.index.get_loc(date)
                    
                    # For rebalance_period=1, we rebalance daily, so we calculate return for the next single day
                    # This is the return earned from the portfolio constructed on the rebalance date
                    end_idx = min(
                        date_idx + self.rebalance_period + 1, len(factor_df.index)
                    )

                    # Only iterate through groups that actually exist
                    actual_groups = groups.dropna().unique()
                    for group_num in actual_groups:
                        # Find assets in this group
                        group_assets = groups[groups == group_num].index

                        if len(group_assets) > 0 and date_idx + 1 < len(daily_ret_df):
                            # Calculate weighted return for this group based on weight_method
                            # Intersect group_assets with available columns in daily_ret_df
                            available_assets = daily_ret_df.columns.intersection(
                                group_assets
                            )
                            if len(available_assets) > 0:
                                # Initialize portfolio value series for this group if not exists
                                if group_num not in all_group_portfolio_values:
                                    all_group_portfolio_values[group_num] = pd.Series(
                                        dtype=float
                                    )

                                # Get previous portfolio value (last value from previous period)
                                if len(all_group_portfolio_values[group_num]) > 0:
                                    prev_portfolio_value = all_group_portfolio_values[
                                        group_num
                                    ].iloc[-1]
                                else:
                                    prev_portfolio_value = 1.0

                                # Calculate current weights (equal weighting)
                                current_weights = pd.Series(
                                    1.0 / len(available_assets), index=available_assets
                                )

                                # Calculate turnover and apply fee
                                if (
                                    group_num in all_group_prev_weights
                                    and group_num in all_group_prev_assets
                                ):
                                    prev_weights = all_group_prev_weights[group_num]
                                    prev_assets = all_group_prev_assets[group_num]

                                    # Create aligned weight series
                                    all_assets = list(
                                        set(list(prev_assets) + list(available_assets))
                                    )
                                    prev_weights_aligned = pd.Series(
                                        0.0, index=all_assets
                                    )
                                    current_weights_aligned = pd.Series(
                                        0.0, index=all_assets
                                    )

                                    for asset in prev_assets:
                                        if asset in prev_weights.index:
                                            prev_weights_aligned[asset] = prev_weights[
                                                asset
                                            ]

                                    for asset in available_assets:
                                        if asset in current_weights.index:
                                            current_weights_aligned[asset] = (
                                                current_weights[asset]
                                            )

                                    # Calculate turnover
                                    turnover = (
                                        current_weights_aligned - prev_weights_aligned
                                    ).abs().sum() / 2.0

                                    # Apply fee based on turnover
                                    fee_amount = (
                                        prev_portfolio_value * self.fee * turnover
                                    )
                                    portfolio_value = prev_portfolio_value - fee_amount
                                else:
                                    portfolio_value = prev_portfolio_value

                                # Store current weights and assets for next rebalancing
                                all_group_prev_weights[group_num] = current_weights
                                all_group_prev_assets[group_num] = available_assets

                                # Calculate returns for each day in the period
                                for day_idx in range(date_idx + 1, end_idx):
                                    if day_idx < len(daily_ret_df.index):
                                        day_date = daily_ret_df.index[day_idx]
                                        # Use .loc to get returns for the specific day and available assets
                                        if day_date in daily_ret_df.index:
                                            day_rets = daily_ret_df.loc[
                                                day_date, available_assets
                                            ]
                                            if not day_rets.empty:
                                                # Apply weighting based on weight_method
                                                if self.weight_method == "equal":
                                                    # Equal weighting
                                                    weighted_return = day_rets.mean()
                                                # Future weight methods can be added here:
                                                # elif self.weight_method == "new_method":
                                                #     # Implementation for new weighting method
                                                #     ...
                                                else:
                                                    # Default to equal weighting for any unrecognized method
                                                    weighted_return = day_rets.mean()

                                                if not pd.isna(weighted_return):
                                                    # Update portfolio value
                                                    portfolio_value *= (
                                                        1 + weighted_return
                                                    )

                                                    # Store portfolio value
                                                    all_group_portfolio_values[
                                                        group_num
                                                    ].loc[day_date] = portfolio_value

                                                    # Store the daily return for this group
                                                    if (
                                                        group_num
                                                        not in all_group_daily_returns
                                                    ):
                                                        all_group_daily_returns[
                                                            group_num
                                                        ] = {}
                                                    all_group_daily_returns[group_num][
                                                        day_date
                                                    ] = weighted_return

                            if (
                                group_num in all_group_portfolio_values
                                and len(all_group_portfolio_values[group_num]) > 0
                            ):
                                # Calculate average return for this period
                                period_values = all_group_portfolio_values[
                                    group_num
                                ].values
                                if len(period_values) >= 2:
                                    period_return = (
                                        period_values[-1] / period_values[0]
                                    ) - 1
                                    avg_group_ret_by_period.loc[date, group_num] = (
                                        period_return
                                    )
                                else:
                                    avg_group_ret_by_period.loc[date, group_num] = (
                                        np.nan
                                    )
                            else:
                                avg_group_ret_by_period.loc[date, group_num] = np.nan
                        else:
                            avg_group_ret_by_period.loc[date, group_num] = np.nan

                    # Handle groups that don't exist due to duplicate values or insufficient data
                    # Fill missing groups with NaN for consistency
                    for group_num in range(self.n_groups):
                        if group_num not in actual_groups:
                            avg_group_ret_by_period.loc[date, group_num] = np.nan
                except Exception:
                    # If quantile cut fails due to ties or insufficient data, assign equally
                    n_assets = len(current_factors)
                    if n_assets > 0:
                        assets_per_group = n_assets // self.n_groups
                        extra_assets = n_assets % self.n_groups

                        sorted_factors = current_factors.sort_values()
                        group_assignments = {}

                        start_idx = 0
                        for group_num in range(self.n_groups):
                            end_idx = start_idx + assets_per_group
                            if group_num < extra_assets:
                                end_idx += 1  # Distribute extra assets to first groups

                            group_assets = sorted_factors.iloc[start_idx:end_idx].index
                            for asset in group_assets:
                                group_assignments[asset] = group_num
                            start_idx = end_idx

                        # Calculate returns for each group in the subsequent period
                        date_idx = factor_df.index.get_loc(date)
                        
                        # For rebalance_period=1, we rebalance daily, so we calculate return for the next single day
                        # This is the return earned from the portfolio constructed on the rebalance date
                        end_idx = min(
                            date_idx + self.rebalance_period + 1, len(factor_df.index)
                        )

                        for group_num in range(self.n_groups):
                            group_assets = [
                                asset
                                for asset, group in group_assignments.items()
                                if group == group_num
                            ]

                            if len(group_assets) > 0 and date_idx + 1 < len(
                                daily_ret_df
                            ):
                                # Calculate weighted return for this group based on weight_method
                                # Intersect group_assets with available columns in daily_ret_df
                                available_assets = daily_ret_df.columns.intersection(
                                    group_assets
                                )
                                if len(available_assets) > 0:
                                    # Initialize portfolio value for this group if not exists
                                    if group_num not in all_group_portfolio_values:
                                        all_group_portfolio_values[group_num] = pd.Series(
                                            dtype=float
                                        )

                                    # Get previous portfolio value
                                    if len(all_group_portfolio_values[group_num]) > 0:
                                        prev_portfolio_value = all_group_portfolio_values[
                                            group_num
                                        ].iloc[-1]
                                    else:
                                        prev_portfolio_value = 1.0

                                    # Track previous weights for turnover calculation
                                    prev_group_weights = all_group_prev_weights.get(
                                        group_num, None
                                    )
                                    prev_group_assets = all_group_prev_assets.get(
                                        group_num, None
                                    )

                                    # Calculate current weights (equal weighting)
                                    current_weights = pd.Series(
                                        1.0 / len(available_assets),
                                        index=available_assets,
                                    )

                                    # Calculate turnover and apply fee
                                    if (
                                        prev_group_weights is not None
                                        and prev_group_assets is not None
                                    ):
                                        # Create aligned weight series
                                        all_assets = list(
                                            set(list(prev_group_assets) + list(available_assets))
                                        )
                                        prev_weights_aligned = pd.Series(
                                            0.0, index=all_assets
                                        )
                                        current_weights_aligned = pd.Series(
                                            0.0, index=all_assets
                                        )

                                        for asset in prev_group_assets:
                                            if asset in prev_group_weights.index:
                                                prev_weights_aligned[asset] = (
                                                    prev_group_weights[asset]
                                                )

                                        for asset in available_assets:
                                            if asset in current_weights.index:
                                                current_weights_aligned[asset] = (
                                                    current_weights[asset]
                                                )

                                        # Calculate turnover
                                        turnover = (
                                            current_weights_aligned
                                            - prev_weights_aligned
                                        ).abs().sum() / 2.0

                                        # Apply fee based on turnover
                                        fee_amount = (
                                            prev_portfolio_value * self.fee * turnover
                                        )
                                        portfolio_value = (
                                            prev_portfolio_value - fee_amount
                                        )
                                    else:
                                        portfolio_value = prev_portfolio_value

                                    # Store current weights and assets for next rebalancing
                                    all_group_prev_weights[group_num] = current_weights
                                    all_group_prev_assets[group_num] = available_assets

                                    # Calculate returns for each day in the period
                                    for day_idx in range(date_idx + 1, end_idx):
                                        if day_idx < len(daily_ret_df.index):
                                            day_date = daily_ret_df.index[day_idx]
                                            # Use .loc to get returns for the specific day and available assets
                                            if day_date in daily_ret_df.index:
                                                day_rets = daily_ret_df.loc[
                                                    day_date, available_assets
                                                ]
                                                if not day_rets.empty:
                                                    # Apply weighting based on weight_method
                                                    if self.weight_method == "equal":
                                                        # Equal weighting
                                                        weighted_return = (
                                                            day_rets.mean()
                                                        )
                                                    # Future weight methods can be added here:
                                                    # elif self.weight_method == "new_method":
                                                    #     # Implementation for new weighting method
                                                    #     ...
                                                    else:
                                                        # Default to equal weighting for any unrecognized method
                                                        weighted_return = (
                                                            day_rets.mean()
                                                        )

                                                    if not pd.isna(weighted_return):
                                                        # Update portfolio value
                                                        portfolio_value *= (
                                                            1 + weighted_return
                                                        )

                                                        # Store portfolio value
                                                        all_group_portfolio_values[
                                                            group_num
                                                        ].loc[day_date] = portfolio_value

                                                        # Store the daily return for this group
                                                        if (
                                                            group_num
                                                            not in all_group_daily_returns
                                                        ):
                                                            all_group_daily_returns[
                                                                group_num
                                                            ] = {}
                                                        all_group_daily_returns[
                                                            group_num
                                                        ][day_date] = weighted_return

                                    if (
                                        group_num in all_group_portfolio_values
                                        and len(all_group_portfolio_values[group_num])
                                        > 0
                                    ):
                                        # Calculate average return for this period
                                        period_values = list(
                                            all_group_portfolio_values[
                                                group_num
                                            ].values
                                        )
                                        if len(period_values) >= 2:
                                            period_return = (
                                                period_values[-1] / period_values[0]
                                            ) - 1
                                            avg_group_ret_by_period.loc[
                                                date, group_num
                                            ] = period_return
                                        else:
                                            avg_group_ret_by_period.loc[
                                                date, group_num
                                            ] = np.nan
                                    else:
                                        avg_group_ret_by_period.loc[date, group_num] = (
                                            np.nan
                                        )
                            else:
                                avg_group_ret_by_period.loc[date, group_num] = np.nan
            else:
                # Skip rebalance dates with insufficient assets - fill all groups with NaN
                for group_num in range(self.n_groups):
                    avg_group_ret_by_period.loc[date, group_num] = np.nan

        # Calculate average group returns along the time axis (average across all rebalance periods for each group)
        avg_group_ret = avg_group_ret_by_period.mean(
            axis=0
        )  # Average across time axis (axis=0)

        # Convert all_group_daily_returns to a DataFrame for visualization
        # all_group_daily_returns now only contains actual return data (int keys)
        group_returns_data = all_group_daily_returns

        if group_returns_data:
            # Get all unique dates across all groups
            all_dates = set()
            for group_daily_returns in group_returns_data.values():
                all_dates.update(group_daily_returns.keys())
            all_dates = sorted(list(all_dates))

            # Create DataFrame for group daily returns with correct column names
            group_nums = sorted([k for k in group_returns_data.keys() if isinstance(k, (int, np.integer))])
            avg_group_daily_ret = pd.DataFrame(
                index=all_dates, columns=group_nums
            )
            for group_num, group_daily_returns in group_returns_data.items():
                if isinstance(group_num, (int, np.integer)):
                    for date, ret in group_daily_returns.items():
                        avg_group_daily_ret.loc[date, group_num] = ret

            # Calculate cumulative returns for each group (using same method as portfolio)
            if cumprod:
                avg_group_cum_ret = (1 + avg_group_daily_ret).cumprod() - 1
            else:
                avg_group_cum_ret = avg_group_daily_ret.cumsum()
        else:
            avg_group_daily_ret = pd.DataFrame()
            avg_group_cum_ret = pd.DataFrame()

        return {
            "daily_ret": daily_ret_df,
            "cum_ret": cum_ret_df,
            "portfolio_daily_ret": avg_daily_returns,
            "portfolio_cum_ret": avg_cum_ret,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "rank_ic": overall_rank_ic,
            "rank_ic_time_series": rank_ic_time_series,
            "cumulative_rank_ic": cumulative_rank_ic,
            "avg_group_ret": avg_group_ret,
            "avg_group_daily_ret": avg_group_daily_ret,
            "avg_group_cum_ret": avg_group_cum_ret,
        }

    def run(self):
        r"""
        Run back-test. The back-testing steps include:
        1. Compute daily returns for each asset.
        2. Compute daily returns for the portfolio.
        3. Compute back-test metrics.
        4. Plot results if required.
        """
        if self.metrics_path is None:
            os.makedirs("metrics_result", exist_ok=True)
            self.metrics_path = os.path.join(os.getcwd(), "metrics_result")

        if self.figures_path is None:
            os.makedirs("figures_result", exist_ok=True)
            self.figures_path = os.path.join(os.getcwd(), "figures_result")

        # Compute the metrics
        metrics_dict = self.compute_metrics(
            self.price_df, self.factor_df, cumprod=self.cumprod
        )

        # Store metrics as instance variables
        self.daily_ret = metrics_dict["daily_ret"]
        self.cum_ret = metrics_dict["cum_ret"]
        self.portfolio_daily_ret = metrics_dict["portfolio_daily_ret"]
        self.portfolio_cum_ret = metrics_dict["portfolio_cum_ret"]
        self.sharpe_ratio = metrics_dict["sharpe_ratio"]
        self.sortino_ratio = metrics_dict["sortino_ratio"]
        self.calmar_ratio = metrics_dict["calmar_ratio"]
        self.max_drawdown = metrics_dict["max_drawdown"]
        self.win_rate = metrics_dict["win_rate"]
        self.rank_ic = metrics_dict["rank_ic"]
        self.rank_ic_time_series = metrics_dict["rank_ic_time_series"]
        self.cumulative_rank_ic = metrics_dict["cumulative_rank_ic"]
        self.avg_group_ret = metrics_dict["avg_group_ret"]
        self.avg_group_daily_ret = metrics_dict["avg_group_daily_ret"]
        self.avg_group_cum_ret = metrics_dict["avg_group_cum_ret"]

        # Save comprehensive metrics to file
        import json

        # Calculate additional metrics
        # Overall portfolio metrics (using portfolio returns with fees)
        overall_daily_returns = self.portfolio_daily_ret
        overall_cumulative_return = self.portfolio_cum_ret

        # Annual return calculation
        if len(overall_daily_returns) > 0:
            trading_days = len(overall_daily_returns)
            total_return = (
                overall_cumulative_return.iloc[-1]
                if len(overall_cumulative_return) > 0
                else 0
            )
            if self.cumprod:
                annual_return = (
                    (1 + total_return) ** (252 / trading_days) - 1
                    if trading_days > 0
                    else 0
                )
            else:
                annual_return = (
                    total_return * (252 / trading_days) if trading_days > 0 else 0
                )
        else:
            annual_return = 0

        # Calculate group-specific metrics
        group_metrics = {}
        if (
            hasattr(self, "avg_group_daily_ret")
            and self.avg_group_daily_ret is not None
            and not self.avg_group_daily_ret.empty
        ):
            for group_num in self.avg_group_daily_ret.columns:
                group_daily_rets = self.avg_group_daily_ret[group_num].dropna()

                if len(group_daily_rets) > 0:
                    # Group cumulative return
                    if self.cumprod:
                        group_cum_ret = (1 + group_daily_rets).cumprod() - 1
                    else:
                        group_cum_ret = group_daily_rets.cumsum()

                    group_total_return = (
                        group_cum_ret.iloc[-1] if len(group_cum_ret) > 0 else 0
                    )

                    # Group annual return
                    trading_days = len(group_daily_rets)
                    if self.cumprod:
                        group_annual_return = (
                            (1 + group_total_return) ** (252 / trading_days) - 1
                            if trading_days > 0
                            else 0
                        )
                    else:
                        group_annual_return = (
                            group_total_return * (252 / trading_days)
                            if trading_days > 0
                            else 0
                        )

                    # Group Sharpe ratio
                    group_mean = group_daily_rets.mean()
                    group_std = group_daily_rets.std()
                    if group_std == 0 or pd.isna(group_std) or pd.isna(group_mean):
                        group_sharpe = 0.0
                    else:
                        group_sharpe = (group_mean / group_std) * np.sqrt(252)

                    # Group Sortino ratio
                    group_negative_returns = group_daily_rets[group_daily_rets < 0]
                    if len(group_negative_returns) > 0:
                        group_negative_std = group_negative_returns.std()
                        if (
                            group_negative_std == 0
                            or pd.isna(group_negative_std)
                            or pd.isna(group_mean)
                        ):
                            group_sortino = 0.0
                        else:
                            group_sortino = (group_mean / group_negative_std) * np.sqrt(
                                252
                            )
                    else:
                        group_sortino = group_sharpe

                    # Group Calmar ratio
                    group_min = group_daily_rets.min()
                    if group_min == 0 or pd.isna(group_min) or pd.isna(group_mean):
                        group_calmar = 0.0
                    else:
                        group_calmar = (group_mean / abs(group_min)) * np.sqrt(252)

                    # Group max drawdown
                    if self.cumprod:
                        group_running_max = (1 + group_cum_ret).cummax()
                        group_drawdown = (1 + group_cum_ret) / group_running_max - 1
                    else:
                        group_running_max = group_cum_ret.expanding().max()
                        group_drawdown = group_cum_ret - group_running_max

                    if group_drawdown.empty or pd.isna(group_drawdown.min()):
                        group_max_drawdown = 0.0
                    else:
                        group_max_drawdown = group_drawdown.min()

                    # Group win rate
                    if group_daily_rets.empty or pd.isna((group_daily_rets > 0).mean()):
                        group_win_rate = 0.0
                    else:
                        group_win_rate = (group_daily_rets > 0).mean()

                    # Group Rank IC (use overall rank_ic as approximation)
                    group_rank_ic = self.rank_ic if hasattr(self, "rank_ic") else 0.0

                    group_metrics[f"group_{group_num}"] = {
                        "annual_return": float(group_annual_return),
                        "cumulative_return": float(group_total_return),
                        "sharpe_ratio": float(group_sharpe),
                        "sortino_ratio": float(group_sortino),
                        "calmar_ratio": float(group_calmar),
                        "max_drawdown": float(group_max_drawdown),
                        "win_rate": float(group_win_rate),
                        "rank_ic": float(group_rank_ic),
                    }

        # Prepare comprehensive metrics dictionary
        comprehensive_metrics = {
            "overall": {
                "sharpe_ratio": float(self.sharpe_ratio),
                "sortino_ratio": float(self.sortino_ratio),
                "calmar_ratio": float(self.calmar_ratio),
                "max_drawdown": float(self.max_drawdown),
                "win_rate": float(self.win_rate),
                "rank_ic": float(self.rank_ic),
                "annual_return": float(annual_return),
                "cumulative_return": (
                    float(overall_cumulative_return.iloc[-1])
                    if len(overall_cumulative_return) > 0
                    else 0.0
                ),
            },
            "groups": group_metrics,
            "metadata": {
                "rebalance_period": self.rebalance_period,
                "n_groups": self.n_groups,
                "weight_method": self.weight_method,
                "fee": float(self.fee),
                "cumprod": self.cumprod,
                "trading_days": len(overall_daily_returns),
            },
        }

        with open(os.path.join(self.metrics_path, "backtest_metrics.json"), "w") as f:
            json.dump(comprehensive_metrics, f, indent=4)

        # Plot results if required
        if self.need_plot:
            self.plot()

        return metrics_dict

    def plot(self):
        self.plot_ret()
        self.plot_ic()
        self.plot_avg_group_ret()
        self.plot_vis_summary()

    def plot_ret(self):
        r"""
        Plot cumulative and daily returns.
        """
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))

        # Plot cumulative returns
        # Plot overall portfolio return with bold red line first
        ax[0].plot(
            self.portfolio_cum_ret.index,
            self.portfolio_cum_ret,
            label="Overall Portfolio Return",
            color="red",
            linewidth=3,
            linestyle="-",
            alpha=0.8,
        )

        # Plot group returns if available
        if (
            hasattr(self, "avg_group_cum_ret")
            and self.avg_group_cum_ret is not None
            and not self.avg_group_cum_ret.empty
        ):
            for group_num in self.avg_group_cum_ret.columns:
                ax[0].plot(
                    self.avg_group_cum_ret.index,
                    self.avg_group_cum_ret[group_num],
                    label=f"Group {group_num} Cumulative Return",
                    alpha=0.7,
                )

        ax[0].set_title("Cumulative Portfolio Returns by Group")
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel("Cumulative Return")
        ax[0].legend()
        ax[0].grid(True)

        # Plot daily returns
        # Plot overall daily return with bold red line first
        ax[1].plot(
            self.portfolio_daily_ret.index,
            self.portfolio_daily_ret,
            label="Overall Daily Return",
            color="red",
            linewidth=2,
            linestyle="-",
            alpha=0.8,
        )

        # Plot group daily returns if available
        if (
            hasattr(self, "avg_group_daily_ret")
            and self.avg_group_daily_ret is not None
            and not self.avg_group_daily_ret.empty
        ):
            for group_num in self.avg_group_daily_ret.columns:
                ax[1].plot(
                    self.avg_group_daily_ret.index,
                    self.avg_group_daily_ret[group_num],
                    label=f"Group {group_num} Daily Return",
                    alpha=0.7,
                )

        ax[1].set_title("Daily Portfolio Returns by Group")
        ax[1].set_xlabel("Date")
        ax[1].set_ylabel("Daily Return")
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "returns.png"))
        plt.close()

    def plot_ic(self, plot_type="time_series"):
        r"""
        Plot information-coefficient time series (Rank IC and Cumulative Rank IC) or histogram.

        Parameters
        ----------
        plot_type : str
            Type of plot to generate. Options: "time_series" (default) or "histogram".
        """
        if plot_type == "time_series":
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot Rank IC and Cumulative Rank IC on the same graph
            rank_ic_available = (
                hasattr(self, "rank_ic_time_series")
                and self.rank_ic_time_series is not None
                and len(self.rank_ic_time_series) > 0
            )
            cumulative_rank_ic_available = (
                hasattr(self, "cumulative_rank_ic")
                and self.cumulative_rank_ic is not None
                and len(self.cumulative_rank_ic) > 0
            )

            if rank_ic_available:
                ax.plot(
                    self.rank_ic_time_series.index,
                    self.rank_ic_time_series.values,
                    linewidth=1.0,
                    label="Rank IC (Spearman)",
                    alpha=0.7,
                    color="blue",
                )

            if cumulative_rank_ic_available:
                # Scale cumulative rank ic for better visualization if needed
                ax_twin = ax.twinx()  # Create secondary y-axis
                ax_twin.plot(
                    self.cumulative_rank_ic.index,
                    self.cumulative_rank_ic.values,
                    linewidth=2,
                    label="Cumulative Rank IC",
                    color="red",
                    alpha=0.8,
                )

                # Set labels for both axes
                ax.set_ylabel("Rank IC", color="blue")
                ax_twin.set_ylabel("Cumulative Rank IC", color="red")
            elif rank_ic_available:
                # If only Rank IC is available, use single y-axis
                ax.set_ylabel("Rank IC")

            if rank_ic_available or cumulative_rank_ic_available:
                ax.set_title("Rank IC and Cumulative Rank IC Over Time")
                ax.set_xlabel("Date")
        elif plot_type == "histogram":
            # Plot histogram of Rank IC values
            fig, ax = plt.subplots(figsize=(12, 8))

            rank_ic_available = (
                hasattr(self, "rank_ic_time_series")
                and self.rank_ic_time_series is not None
                and len(self.rank_ic_time_series) > 0
            )

            if rank_ic_available:
                ax.hist(
                    self.rank_ic_time_series.values,
                    bins=20,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                )

                ax.set_title("Histogram of Rank IC Values")
                ax.set_xlabel("Rank IC (Spearman)")
                ax.set_ylabel("Frequency")
                ax.grid(True, axis="y")

                # Add mean and median lines
                mean_ic = self.rank_ic_time_series.mean()
                median_ic = self.rank_ic_time_series.median()
                ax.axvline(
                    mean_ic,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_ic:.4f}",
                )
                ax.axvline(
                    median_ic,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Median: {median_ic:.4f}",
                )
                ax.legend()
            else:
                print("No Rank IC time series data available for histogram.")
            ax.grid(True)

            # Create legend combining both plots
            lines_1, labels_1 = ax.get_legend_handles_labels()
            if cumulative_rank_ic_available:
                lines_2, labels_2 = ax_twin.get_legend_handles_labels()
                ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
            else:
                ax.legend(loc="upper left")

            # Add horizontal line at y=0
            ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        else:
            # Fallback: calculate Rank IC and Cumulative Rank IC if not available
            if hasattr(self, "factor_df") and hasattr(self, "price_df"):
                # Calculate daily returns
                daily_returns = self.price_df.pct_change().dropna()
                rank_ic_series = []
                dates = []

                for date in self.factor_df.index:
                    if date in daily_returns.index:
                        # Calculate Rank IC at this date
                        factors_at_date = self.factor_df.loc[date].dropna()
                        returns_next_day = (
                            daily_returns.loc[date]
                            if date in daily_returns.index
                            else None
                        )

                        if returns_next_day is not None:
                            returns_aligned = factors_at_date.index.intersection(
                                returns_next_day.index
                            )
                            if (
                                len(returns_aligned) > 1
                            ):  # Need at least 2 points for correlation
                                aligned_factors = factors_at_date[returns_aligned]
                                aligned_returns = returns_next_day[returns_aligned]

                                # Calculate Rank IC (correlation of ranks)
                                rank_ic_val = aligned_factors.rank().corr(
                                    aligned_returns, method="spearman"
                                )

                                if not pd.isna(rank_ic_val):
                                    rank_ic_series.append(rank_ic_val)
                                    dates.append(date)

                if dates and rank_ic_series:
                    ax.plot(
                        dates,
                        rank_ic_series,
                        linewidth=1.0,
                        label="Rank IC (Spearman)",
                        alpha=0.7,
                        color="blue",
                    )

                    # Calculate cumulative rank ic
                    rank_ic_series_pd = pd.Series(
                        data=rank_ic_series, index=dates, name="Rank_IC"
                    )
                    cumulative_rank_ic_series = rank_ic_series_pd.cumsum()
                    ax_twin = ax.twinx()
                    ax_twin.plot(
                        cumulative_rank_ic_series.index,
                        cumulative_rank_ic_series.values,
                        linewidth=2,
                        label="Cumulative Rank IC",
                        color="red",
                        alpha=0.8,
                    )

                    ax.set_ylabel("Rank IC", color="blue")
                    ax_twin.set_ylabel("Cumulative Rank IC", color="red")

                    ax.set_title("Rank IC and Cumulative Rank IC Over Time")
                    ax.set_xlabel("Date")
                    ax.grid(True)

                    # Create legend combining both plots
                    lines_1, labels_1 = ax.get_legend_handles_labels()
                    lines_2, labels_2 = ax_twin.get_legend_handles_labels()
                    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

                    # Add horizontal line at y=0
                    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, "ic.png"))
        plt.close()

    def plot_avg_group_ret(self):
        r"""
        Plot average returns by group.
        """
        if hasattr(self, "avg_group_ret") and self.avg_group_ret is not None:
            fig, ax = plt.subplots(figsize=(12, 6))

            # avg_group_ret is now a Series with group numbers as index and average returns as values
            groups = self.avg_group_ret.index
            avg_returns = self.avg_group_ret.values

            # Plot bar chart since we have one average value per group
            bars = ax.bar(groups.astype(str), avg_returns, alpha=0.7)

            # Add value labels on top of bars
            for bar, value in zip(bars, avg_returns):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                )

            ax.set_title("Average Returns by Group")
            ax.set_xlabel("Group")
            ax.set_ylabel("Average Return")
            ax.grid(True, axis="y")

            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_path, "avg_group_returns.png"))
            plt.close()

    def plot_vis_summary(self):
        r"""
        Create comprehensive visualization with three distinct sections arranged vertically:
        - Top: Average returns by group (bar chart)
        - Middle: Cumulative returns by group and overall portfolio
        - Bottom: Rank IC (bars) and cumulative Rank IC (line)
        Save as vis_summary.png.
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 2.5, 1], hspace=0.4)

        ax_top = fig.add_subplot(gs[0])
        ax_mid = fig.add_subplot(gs[1])
        ax_bot = fig.add_subplot(gs[2])

        if hasattr(self, "avg_group_ret") and self.avg_group_ret is not None:
            groups = self.avg_group_ret.index
            avg_returns = self.avg_group_ret.values
            bars = ax_top.bar(
                [str(g) for g in groups],
                avg_returns,
                alpha=0.7,
                edgecolor="black",
                linewidth=1.2,
            )
            for bar, value in zip(bars, avg_returns):
                height = bar.get_height()
                ax_top.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            ax_top.set_title("Average Returns by Group", fontsize=14, fontweight="bold")
            ax_top.set_xlabel("Group", fontsize=11)
            ax_top.set_ylabel("Average Return", fontsize=11)
            ax_top.grid(True, axis="y", alpha=0.3)

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        if (
            hasattr(self, "avg_group_cum_ret")
            and self.avg_group_cum_ret is not None
            and not self.avg_group_cum_ret.empty
        ):
            for idx, group_num in enumerate(self.avg_group_cum_ret.columns):
                ax_mid.plot(
                    self.avg_group_cum_ret.index,
                    self.avg_group_cum_ret[group_num],
                    label=f"Group {group_num}",
                    color=colors[idx % len(colors)],
                    linewidth=1.5,
                    alpha=0.8,
                )

            if (
                hasattr(self, "portfolio_cum_ret")
                and self.portfolio_cum_ret is not None
            ):
                ax_mid.plot(
                    self.portfolio_cum_ret.index,
                    self.portfolio_cum_ret.values,
                    label="Overall Portfolio",
                    color="red",
                    linewidth=2.5,
                    linestyle="-",
                    alpha=0.9,
                )

            ax_mid.set_title(
                "Cumulative Returns by Group and Overall Portfolio",
                fontsize=14,
                fontweight="bold",
            )
            ax_mid.set_xlabel("Date", fontsize=11)
            ax_mid.set_ylabel("Cumulative Return", fontsize=11)
            ax_mid.legend(loc="upper left", fontsize=10)
            ax_mid.grid(True, alpha=0.3)

            if len(self.avg_group_cum_ret.index) > 0:
                from matplotlib.dates import MonthLocator, DateFormatter

                ax_mid.xaxis.set_major_locator(MonthLocator(interval=3))
                ax_mid.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
                plt.setp(ax_mid.xaxis.get_majorticklabels(), rotation=45, ha="right")

        if (
            hasattr(self, "rank_ic_time_series")
            and self.rank_ic_time_series is not None
            and len(self.rank_ic_time_series) > 0
        ):
            x_dates = self.rank_ic_time_series.index
            x_numeric = range(len(x_dates))

            bars = ax_bot.bar(
                x_numeric,
                self.rank_ic_time_series.values,
                color="blue",
                alpha=0.6,
                width=0.8,
                label="Rank IC",
            )

            if (
                hasattr(self, "cumulative_rank_ic")
                and self.cumulative_rank_ic is not None
                and len(self.cumulative_rank_ic) > 0
            ):
                ax_bot_twin = ax_bot.twinx()
                ax_bot_twin.plot(
                    x_numeric,
                    self.cumulative_rank_ic.values,
                    color="red",
                    linewidth=2,
                    label="Cumulative Rank IC",
                    alpha=0.9,
                )
                ax_bot_twin.set_ylabel("Cumulative Rank IC", color="red", fontsize=11)
                ax_bot_twin.tick_params(axis="y", labelcolor="red")

                lines_1, labels_1 = ax_bot.get_legend_handles_labels()
                lines_2, labels_2 = ax_bot_twin.get_legend_handles_labels()
                ax_bot.legend(
                    lines_1 + lines_2,
                    labels_1 + labels_2,
                    loc="upper left",
                    fontsize=10,
                )
            else:
                ax_bot.legend(loc="upper left", fontsize=10)

            ax_bot.axhline(y=0, color="black", linestyle="--", alpha=0.3)
            ax_bot.set_title(
                "Rank IC and Cumulative Rank IC Over Time",
                fontsize=14,
                fontweight="bold",
            )
            ax_bot.set_xlabel("Date", fontsize=11)
            ax_bot.set_ylabel("Rank IC", color="blue", fontsize=11)
            ax_bot.tick_params(axis="y", labelcolor="blue")
            ax_bot.grid(True, alpha=0.3)

            if len(x_dates) > 0:
                tick_interval = max(1, len(x_dates) // 10)
                ax_bot.set_xticks(x_numeric[::tick_interval])
                ax_bot.set_xticklabels(
                    [str(d)[:10] for d in x_dates[::tick_interval]],
                    rotation=45,
                    ha="right",
                )

        # plt.suptitle(
        #     "Backtest Summary Visualization", fontsize=16, fontweight="bold", y=0.995
        # )
        plt.savefig(
            os.path.join(self.figures_path, "vis_summary.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    # Example usage of the backtest framework
    # Load example data
    import os

    if os.path.exists("example_price.csv") and os.path.exists("example_factors.csv"):
        price_df = pd.read_csv("example_price.csv", index_col=0)
        factor_df = pd.read_csv("example_factors.csv", index_col=0)
        market_cap_df = pd.read_csv("example_market_cap.csv", index_col=0)

        # Convert index to datetime
        price_df.index = pd.to_datetime(price_df.index)
        factor_df.index = pd.to_datetime(factor_df.index)
        market_cap_df.index = pd.to_datetime(market_cap_df.index)

        bt = Backtest(
            factor_df=factor_df,
            price_df=price_df,
            fee=3 * 1e-4,
            rebalance_period=5,
            n_groups=5,
            weight_method="market_cap",
            market_cap_df=market_cap_df,
            need_plot=True,
            need_preprocess=True,
        )
        bt.run()
        print("Backtest completed successfully!")
