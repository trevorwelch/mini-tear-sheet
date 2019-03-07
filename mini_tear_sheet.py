import pandas as pd
import numpy as np

def create_tear_sheet(strategy_returns, benchmark_returns, plot=False):
    strategy_returns_stats = [annual_return(strategy_returns), 
                            annual_volatility(strategy_returns),
                             value_at_risk(strategy_returns),
                             max_drawdown(strategy_returns),
                              cagr(strategy_returns),
                              cagr_over_mdd(strategy_returns)/100,
                              sharpe_ratio(strategy_returns)/100
                             ]
    benchmark_returns_stats = [annual_return(benchmark_returns), 
                            annual_volatility(benchmark_returns),
                             value_at_risk(benchmark_returns),
                             max_drawdown(benchmark_returns),
                              cagr(benchmark_returns),
                              cagr_over_mdd(benchmark_returns)/100,
                              sharpe_ratio(benchmark_returns)/100
                             ]

    mini_tear_sheet = pd.DataFrame()

    mini_tear_sheet['strategy'] = [round(x*100,2) for x in strategy_returns_stats]

    mini_tear_sheet['benchmark'] = [round(x*100,2) for x in benchmark_returns_stats]

    mini_tear_sheet.index = ['Annual Return',
                            'Annual Volatility',
                            'Value at Risk',
                            'Max Drawdown',
                            'CAGR',
                            'MAR Ratio',
                            'Sharpe Ratio']

    if plot == True:
        ax1 = plot_rolling_sharpe(strategy_returns, benchmark_returns)

    return mini_tear_sheet

def annual_return(returns):
    """
    Determines the mean annual growth rate of returns. This is equivilent
    to the compound annual growth rate.

    Resample your returns to daily returns before feeding to this function.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    Returns
    -------
    annual_return : float
        Annual Return as CAGR (Compounded Annual Growth Rate).

    """

    if len(returns) < 1:
        return np.nan

    num_years = len(returns) / 252
    # Pass array to ensure index -1 looks up successfully.
    ending_value = cum_returns_final(returns, starting_value=1)

    return ending_value ** (1 / num_years) - 1  


def cum_returns_final(returns, starting_value=0):
    """
    Compute total returns from simple returns.

    Parameters
    ----------
    returns : pd.DataFrame, pd.Series, or np.ndarray
       Noncumulative simple returns of one or more timeseries.
    starting_value : float, optional
       The starting returns.

    Returns
    -------
    total_returns : pd.Series, np.ndarray, or float
        If input is 1-dimensional (a Series or 1D numpy array), the result is a
        scalar.

        If input is 2-dimensional (a DataFrame or 2D numpy array), the result
        is a 1D array containing cumulative returns for each column of input.
    """
    if len(returns) == 0:
        return np.nan

    if isinstance(returns, pd.DataFrame):
        result = (returns + 1).prod()
    else:
        result = np.nanprod(returns + 1, axis=0)

    if starting_value == 0:
        result -= 1
    else:
        result *= starting_value

    return result    

def cum_returns(returns, starting_value=100000, out=None):
    """
    Compute cumulative returns from simple returns.

    Parameters
    ----------
    returns : pd.Series, np.ndarray, or pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example::

            2015-07-16   -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902

         - Also accepts two dimensional data. In this case, each column is
           cumulated.

    starting_value : float, optional
       The starting returns.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    cumulative_returns : array-like
        Series of cumulative returns.
    """
    if len(returns) < 1:
        return returns.copy()

    nanmask = np.isnan(returns)
    if np.any(nanmask):
        returns = returns.copy()
        returns[nanmask] = 0

    allocated_output = out is None
    if allocated_output:
        out = np.empty_like(returns)

    np.add(returns, 1, out=out)
    out.cumprod(axis=0, out=out)

    if starting_value == 0:
        np.subtract(out, 1, out=out)
    else:
        np.multiply(out, starting_value, out=out)

    if allocated_output:
        if returns.ndim == 1 and isinstance(returns, pd.Series):
            out = pd.Series(out, index=returns.index)
        elif isinstance(returns, pd.DataFrame):
            out = pd.DataFrame(
                out, index=returns.index, columns=returns.columns,
            )

    return out    

def annual_volatility(returns,
                      alpha=2.0,
                      out=None):
    """
    Determines the annual volatility of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.

    alpha : float, optional
        Scaling relation (Levy stability exponent).
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    annual_volatility : float
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    np.nanstd(returns, ddof=1, axis=0, out=out)
    out = np.multiply(out, 252 ** (1.0 / alpha), out=out)
    if returns_1d:
        out = out.item()
    return out

def value_at_risk(returns, sigma=2.0):
    """
    Get value at risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.

    sigma : float, optional
        Standard deviations of VaR, default 2.
    """

    value_at_risk = returns.mean() - sigma * returns.std()

    return value_at_risk    

def max_drawdown(returns, out=None):
    """
    Determines the maximum drawdown of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    max_drawdown : float

    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    returns_1d = returns.ndim == 1

    if len(returns) < 1:
        out[()] = np.nan
        if returns_1d:
            out = out.item()
        return out

    returns_array = np.asanyarray(returns)

    cumulative = np.empty(
        (returns.shape[0] + 1,) + returns.shape[1:],
        dtype='float64',
    )
    cumulative[0] = start = 100
    cum_returns(returns_array, starting_value=start, out=cumulative[1:])

    max_return = np.fmax.accumulate(cumulative, axis=0)

    np.nanmin((cumulative - max_return) / max_return, axis=0, out=out)
    if returns_1d:
        out = out.item()
    elif allocated_output and isinstance(returns, pd.DataFrame):
        out = pd.Series(out)

    return out    

def sharpe_ratio(returns,
                 risk_free=0.00,
                 out=None):
    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~empyrical.stats.cum_returns`.
    risk_free : int, float
        Constant risk-free return throughout the period.

    out : array-like, optional
        Array to use as output buffer.
        If not passed, a new array will be created.

    Returns
    -------
    sharpe_ratio : float
        nan if insufficient length of returns or if if adjusted returns are 0.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.

    """
    allocated_output = out is None
    if allocated_output:
        out = np.empty(returns.shape[1:])

    return_1d = returns.ndim == 1

    if len(returns) < 2:
        out[()] = np.nan
        if return_1d:
            out = out.item()
        return out

    returns_risk_adj = np.asanyarray(_adjust_returns(returns, risk_free))

    np.multiply(
        np.divide(
            np.nanmean(returns_risk_adj, axis=0),
            np.nanstd(returns_risk_adj, ddof=1, axis=0),
            out=out,
        ),
        np.sqrt(252),
        out=out,
    )
    if return_1d:
        out = out.item()

    return out


def _adjust_returns(returns, adjustment_factor):
    """
    Returns the returns series adjusted by adjustment_factor. Optimizes for the
    case of adjustment_factor being 0 by returning returns itself, not a copy!

    Parameters
    ----------
    returns : pd.Series or np.ndarray
    adjustment_factor : pd.Series or np.ndarray or float or int

    Returns
    -------
    adjusted_returns : array-like
    """
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns
    return returns - adjustment_factor


def cagr(returns):
    """
    Compound Annual Growth Rate, given an initial and a final value for an investment,
    as well as the time elapsed (in years or fractions of years)
    """
    initial = 100000
    final = round(calc_end_value_from_returns(returns),2)

    n_days = (returns.index[-1] - returns.index[0]).days

    years = n_days / 365

    if years == 0:
        raise Exception('The time period cannot be zero')

    return (final / initial) ** (1.0 / years) - 1


def calc_end_value_from_returns(returns):
    cumulative = np.empty(
        (returns.shape[0] + 1,) + returns.shape[1:],
        dtype='float64',
    )
    cumulative[0] = start = 100000
    return cum_returns(returns.values, starting_value=start, out=cumulative[1:])[-1]


def rolling_sharpe(returns, rolling_sharpe_window):
    """
    Determines the rolling Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    rolling_sharpe_window : int
        Length of rolling window, in days, over which to compute.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.
    """

    return returns.rolling(rolling_sharpe_window).mean() \
        / returns.rolling(rolling_sharpe_window).std() \
        * np.sqrt(252)


def cagr_over_mdd(returns):
    return (cagr(returns)*100) / (max_drawdown(returns)*100*-1)


### PLOTTING ###

def plot_rolling_sharpe(strategy_returns, benchmark_returns, rolling_window=21 * 6,
                        legend_loc='best', ax=None, **kwargs):
    

    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter

    def two_dec_places(x, pos):
        """
        Adds 1/100th decimal to plot ticks.
        """

        return '%.2f' % x
    """
    Plots the rolling Sharpe ratio versus date.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    rolling_window : int, optional
        The days window over which to compute the sharpe ratio.
    legend_loc : matplotlib.loc, optional
        The location of the legend on the plot.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    fig, ax = plt.subplots(figsize=(15,5))

    y_axis_formatter = FuncFormatter(two_dec_places)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    x_axis_formatter = FuncFormatter(two_dec_places)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))

    rolling_sharpe_ts = rolling_sharpe(
        strategy_returns, rolling_window)
    rolling_sharpe_ts.plot(alpha=.7, lw=3, color='green', ax=ax)

    rolling_sharpe_ts = rolling_sharpe(
        benchmark_returns, rolling_window)
    rolling_sharpe_ts.plot(alpha=.2, lw=3, color='blue', ax=ax)

    ax.set_title('Rolling Sharpe ratio (6-month)')
    ax.axhline(
        rolling_sharpe_ts.mean(),
        color='steelblue',
        linestyle='--',
        lw=3)
    # ax.axhline(0.0, color='black', linestyle='-', lw=3)
    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('')
    ax.legend(['Strategy Sharpe', 'Benchmark Sharpe', 'Strategy Average'],
              loc=legend_loc, frameon=True, framealpha=0.5)
    return ax 