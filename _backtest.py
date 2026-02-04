import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class BackTest:
    def __init__(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        rebalance_period: int,
        n_groups: int,
        weight_method: str = "equal",
        need_preprocess: bool = True,
        need_normalize: bool = True,
        price_threshold: float = 1e6,
        factor_threshold: float = 10,
        need_plot: bool = True,
        metrics_path: str = None,
        figures_path: str = None,
        cumprod: bool = True,
        auto_run: bool = False,
    ):
        r"""
        Back-test engine for factor-based strategies.

        Parameters
        ----------
        factor_df : pd.DataFrame
            Factor values for each asset at each time point.
        price_df : pd.DataFrame
            Close prices for each asset at each time point.
        rebalance_period : int
            Rebalancing frequency (number of periods between portfolio shifts). Day as the unit.
        n_groups : int
            Number of quantile groups into which assets are partitioned.
        weight_method : str
            Weighting scheme applied within each group. Options: "equal", "market_cap". Default is "equal".
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

        Attributes
        ----------
        daily_ret : pd.Series
            Daily portfolio returns.
        cum_ret : pd.Series
            Cumulative portfolio returns.
        sharpe_ratio : float
            Annualized Sharpe ratio.
        sortino_ratio : float
            Annualized Sortino ratio.
        calmar_ratio : float
            Annualized Calmar ratio.
        max_drawdown : float
            Maximum drawdown experienced.
        win_rate : float
            Fraction of positive-return periods.
        ic : float
            Information coefficient (factor vs. forward return).
        rank_ic : float
            Rank information coefficient.
        avg_group_ret : pd.DataFrame
            Average return per group per period.

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

        assert isinstance(
            self.factor_df, pd.DataFrame
        ), "factor_df must be a pandas DataFrame."
        assert isinstance(
            self.price_df, pd.DataFrame
        ), "price_df must be a pandas DataFrame."

        # Validate weight_method with only supported options
        # Allow for extensibility - only validate currently supported methods
        # Future weight methods can be added here as they are implemented
        supported_methods = ["equal"]  # Add new methods to this list as they are implemented
        if weight_method not in supported_methods:
            raise ValueError(f"Invalid weight_method '{weight_method}'. Supported methods: {supported_methods}")

        # Verify that we have at least one stock price column
        assert (
            len(self.price_df.columns) > 0
        ), "At least one stock price column must be provided in price_df."

        # Restrict weight_method to only "equal" since market_cap functionality has been removed
        # Allow for extensibility - only validate currently supported methods
        # Future weight methods can be added here as they are implemented
        supported_methods = ["equal"]  # Add new methods to this list as they are implemented
        if weight_method not in supported_methods:
            raise ValueError(f"Invalid weight_method '{weight_method}'. Supported methods: {supported_methods}")

        if need_preprocess:
            self.preprocess(self.price_df, self.factor_df)
        else:
            self.time_index = self.factor_df.index.intersection(self.price_df.index)
            self.price_df = self.price_df.loc[self.time_index]
            self.factor_df = self.factor_df.loc[self.time_index]

        # self.stocks = factor_df.columns.tolist()
        # assert self.stocks == price_df.columns.to_list(), "factor_df must have the same columns as price_df."
        
        # Automatically run the backtest after initialization
        if self.auto_run:
            self.run()

    def preprocess(
        self,
        price_df: pd.DataFrame,
        factor_df: pd.DataFrame,
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
        """
        if len(factor_df) == 0 or len(price_df) == 0:
            raise ValueError("No data provided.")

        factor_df = factor_df.dropna()
        price_df = price_df.dropna()

        if len(factor_df) == 0 or len(price_df) == 0:
            raise ValueError("No valid data after removing nans.")

        price_df = price_df[(price_df > 0) & (price_df < self.price_threshold)]
        factor_mean = factor_df.mean()
        factor_std = factor_df.std()
        factor_df = factor_df[
            (factor_df > factor_mean - self.factor_threshold * factor_std)
            & (factor_df < factor_mean + self.factor_threshold * factor_std)
        ]

        if len(factor_df) == 0 or len(price_df) == 0:
            raise ValueError("No valid data after removing nans and outliers.")

        if not isinstance(factor_df.index, pd.DatetimeIndex):
            factor_df.index = pd.to_datetime(factor_df.index)
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df.index = pd.to_datetime(price_df.index)

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

    def compute_metrics(
        self,
        price_df: pd.DataFrame,
        factor_df: pd.DataFrame,
        cumprod: bool = True,
    ) -> dict:
        r"""
        Compute back-test metrics. The metrics include:
        daily_ret : pd.Series
            Daily portfolio returns.
        cum_ret : pd.Series
            Cumulative portfolio returns.
        sharpe_ratio : float
            Annualized Sharpe ratio.
        sortino_ratio : float
            Annualized Sortino ratio.
        calmar_ratio : float
            Annualized Calmar ratio.
        max_drawdown : float
            Maximum drawdown experienced.
        win_rate : float
            Fraction of positive-return periods.
        ic : float
            Information coefficient (factor vs. forward return).
        rank_ic : float
            Rank information coefficient.
        avg_group_ret : pd.DataFrame
            Average return per group per period.

        Parameters
        ----------
        price_df : pd.DataFrame
            Close prices for each asset at each time point.
        factor_df : pd.DataFrame
            Factor values for each asset at each time point.
        """
        cumprod = self.cumprod
        daily_ret_df = price_df.pct_change().dropna()
        if cumprod:
            cum_ret_df = (1 + daily_ret_df).cumprod() - 1
        else:
            cum_ret_df = daily_ret_df.cumsum()

        # Calculate portfolio-level metrics considering weight_method
        # Extensible design - add new weight methods in elif clauses
        if self.weight_method == "equal":
            # Equal weighting - average across assets
            avg_daily_returns = daily_ret_df.mean(
                axis=1
            )  # Average return across assets per day
            avg_cum_ret = (
                cum_ret_df.mean(axis=1) if cumprod else cum_ret_df.mean(axis=1)
            )
        # Future weight methods can be added here:
        # elif self.weight_method == "new_method":
        #     # Implementation for new weighting method
        #     ...
        else:
            # Default to equal weighting for any unrecognized method
            avg_daily_returns = daily_ret_df.mean(axis=1)
            avg_cum_ret = (
                cum_ret_df.mean(axis=1)
                if cumprod
                else cum_ret_df.mean(axis=1)
            )

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
        for i, date in enumerate(factor_dates[:-1]):  # Exclude the last date since there's no future return
            if date in daily_ret_df.index:
                # Get factors at current date (time t)
                factors_at_date = factor_df.loc[date].dropna()
                
                # Get returns at the next date (time t+1) - this avoids forward-looking bias
                next_date_idx = i + 1
                if next_date_idx < len(factor_dates) and factor_dates[next_date_idx] in daily_ret_df.index:
                    returns_at_next_date = daily_ret_df.loc[factor_dates[next_date_idx]].dropna()

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
        shifted_return_df = daily_ret_df.shift(-1)  # Shift returns back by 1 so factor t predicts return t+1
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
                    end_idx = min(
                        date_idx + self.rebalance_period, len(factor_df.index)
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
                                group_returns = []
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
                                                    group_returns.append(
                                                        weighted_return
                                                    )
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

                            if group_returns:
                                avg_group_ret_by_period.loc[date, group_num] = np.mean(
                                    group_returns
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
                        end_idx = min(
                            date_idx + self.rebalance_period, len(factor_df.index)
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
                                group_returns = []  # Initialize here to ensure it exists in all code paths
                                if len(available_assets) > 0:
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

                                                    if not pd.isna(
                                                        weighted_return
                                                    ):
                                                        group_returns.append(
                                                            weighted_return
                                                        )
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
                                                        ][
                                                            day_date
                                                        ] = weighted_return

                                if group_returns:
                                    avg_group_ret_by_period.loc[date, group_num] = (
                                        np.mean(group_returns)
                                    )
                                else:
                                    avg_group_ret_by_period.loc[date, group_num] = (
                                        np.nan
                                    )
                            else:
                                avg_group_ret_by_period.loc[date, group_num] = np.nan

        # Calculate average group returns along the time axis (average across all rebalance periods for each group)
        avg_group_ret = avg_group_ret_by_period.mean(
            axis=0
        )  # Average across time axis (axis=0)

        # Convert all_group_daily_returns to a DataFrame for visualization
        if all_group_daily_returns:
            # Get all unique dates across all groups
            all_dates = set()
            for group_daily_returns in all_group_daily_returns.values():
                all_dates.update(group_daily_returns.keys())
            all_dates = sorted(list(all_dates))

            # Create DataFrame for group daily returns
            avg_group_daily_ret = pd.DataFrame(
                index=all_dates, columns=range(len(all_group_daily_returns))
            )
            for group_num, group_daily_returns in all_group_daily_returns.items():
                for date, ret in group_daily_returns.items():
                    avg_group_daily_ret.loc[date, group_num] = ret

            # Calculate cumulative returns for each group
            avg_group_cum_ret = avg_group_daily_ret.fillna(0).cumsum()
        else:
            avg_group_daily_ret = pd.DataFrame()
            avg_group_cum_ret = pd.DataFrame()

        return {
            "daily_ret": daily_ret_df,
            "cum_ret": cum_ret_df,
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

        # Save metrics to file
        import json

        with open(os.path.join(self.metrics_path, "backtest_metrics.json"), "w") as f:
            # Convert series to scalar values by taking mean or appropriate aggregation
            json.dump(
                {
                    "sharpe_ratio": (
                        float(self.sharpe_ratio.mean())
                        if hasattr(self.sharpe_ratio, "mean")
                        else float(self.sharpe_ratio)
                    ),
                    "sortino_ratio": (
                        float(self.sortino_ratio.mean())
                        if hasattr(self.sortino_ratio, "mean")
                        else float(self.sortino_ratio)
                    ),
                    "calmar_ratio": (
                        float(self.calmar_ratio.mean())
                        if hasattr(self.calmar_ratio, "mean")
                        else float(self.calmar_ratio)
                    ),
                    "max_drawdown": (
                        float(self.max_drawdown.mean())
                        if hasattr(self.max_drawdown, "mean")
                        else float(self.max_drawdown)
                    ),
                    "win_rate": (
                        float(self.win_rate.mean())
                        if hasattr(self.win_rate, "mean")
                        else float(self.win_rate)
                    ),
                    "rank_ic": (
                        float(self.rank_ic.mean())
                        if hasattr(self.rank_ic, "mean")
                        else float(self.rank_ic)
                    ),
                },
                f,
            )

        # Plot results if required
        if self.need_plot:
            self.plot_ret()
            self.plot_ic()
            self.plot_avg_group_ret()

        return metrics_dict

    def plot_ret(self):
        r"""
        Plot cumulative and daily returns.
        """
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))

        # Plot cumulative returns
        # Plot overall portfolio return with bold red line first
        ax[0].plot(
            self.cum_ret.index,
            self.cum_ret.mean(axis=1),
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
            self.daily_ret.index,
            self.daily_ret.mean(axis=1),
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
                ax.axvline(mean_ic, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_ic:.4f}")
                ax.axvline(median_ic, color="green", linestyle="--", linewidth=2, label=f"Median: {median_ic:.4f}")
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

if __name__ == "__main__":
    # Example usage of the backtest framework
    # Load example data
    import os
    if os.path.exists("example_price.csv") and os.path.exists("example_factors.csv"):
        price_df = pd.read_csv("example_price.csv", index_col='time')
        factor_df = pd.read_csv("example_factors.csv", index_col='Date')
        
        # Convert index to datetime
        price_df.index = pd.to_datetime(price_df.index)
        factor_df.index = pd.to_datetime(factor_df.index)
        
        # Use only numeric columns for both dataframes
        price_df = price_df.select_dtypes(include=[np.number])
        factor_df = factor_df.select_dtypes(include=[np.number])
        
        # Ensure we have sufficient overlapping dates
        common_dates = price_df.index.intersection(factor_df.index)
        if len(common_dates) > 10:  # Need at least 10 days for meaningful backtest
            price_df = price_df.loc[common_dates]
            factor_df = factor_df.loc[common_dates]
            
            # Limit to a reasonable number of columns to prevent issues
            price_cols = min(5, len(price_df.columns))
            factor_cols = min(3, len(factor_df.columns))
            
            price_df = price_df.iloc[:, :price_cols]
            factor_df = factor_df.iloc[:, :factor_cols]
            
            try:
                bt = BackTest(
                    factor_df=factor_df,
                    price_df=price_df,
                    rebalance_period=5,
                    n_groups=3,
                    weight_method='equal',
                    need_plot=True,
                    need_preprocess=True
                )
                bt.run()
                print("Backtest completed successfully!")
                print(f"Sharpe Ratio: {bt.sharpe_ratio}")
            except Exception as e:
                print(f"Error during backtest: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Insufficient overlapping data for backtest")
    else:
        print("Example files not found. Please ensure 'example_price.csv' and 'example_factors.csv' exist in the directory.")