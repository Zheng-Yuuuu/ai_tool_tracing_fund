"""Latest disclosed top-ten basket: chronological train/validation diagnostic."""
import numpy as np
import pandas as pd
import config as c


def select_by_historical_fit(holdings, stocks, aggregate, signals):
    """Select with a fixed disclosed basket and prices strictly before each signal date.

    Historical returns before report_date may be used to diagnose today's known
    basket. This does not assert that basket was held during the fitting window.
    Report availability still uses report_date, as in the rest of this MVP.
    """
    dates = pd.DatetimeIndex(aggregate.date)
    benchmark = aggregate.set_index('date').ai_return.reindex(dates).to_numpy()
    prices = stocks.pivot(index='date', columns='stock_code', values='daily_return').reindex(dates)
    results = []
    for code, fund in signals.groupby('fund_code'):
        positions = holdings[holdings.fund_code == code]
        reports = sorted(positions.report_date.unique())
        baskets = {}
        for report in reports:
            weights = positions[positions.report_date == report].nlargest(10, 'weight').groupby('stock_code').weight.sum()
            total = weights.sum()
            returns = prices.reindex(columns=weights.index)
            coverage = returns.notna().mul(weights, axis=1).sum(axis=1) / total if total > 0 else pd.Series(0., index=dates)
            baskets[report] = (returns.mul(weights, axis=1).sum(axis=1).to_numpy() / total, coverage.to_numpy())
        for row in fund.itertuples(index=False):
            end = dates.searchsorted(row.date)
            available = [r for r in reports if r <= row.date]
            record = dict(date=row.date, fund_code=code, fit_pass=False, fit_status='数据不足',
                          fit_report_date=pd.NaT, fit_window_end=pd.NaT, min_quote_coverage=np.nan,
                          alpha=np.nan, beta=np.nan, validation_corr=np.nan, oos_r2=np.nan, validation_rmse=np.nan)
            if available and end >= c.FIT_WINDOW:
                report = available[-1]
                basket, coverage = baskets[report]
                start = end - c.FIT_WINDOW
                record.update(fit_report_date=report, fit_window_end=dates[end-1], min_quote_coverage=float(coverage[start:end].min()))
                if record['min_quote_coverage'] < c.FIT_MIN_COVERAGE:
                    record['fit_status'] = '行情覆盖不足'
                elif np.isfinite(benchmark[start:end]).all():
                    record.update(score_fit(basket[start:end], benchmark[start:end], c.FIT_TRAIN_DAYS))
                    record['fit_pass'] = bool(record['beta'] > 0 and record['validation_corr'] >= c.FIT_MIN_CORR and record['oos_r2'] >= c.FIT_MIN_OOS_R2)
                    record['fit_status'] = '通过' if record['fit_pass'] else '未通过'
            results.append(record)
    selected = signals.merge(pd.DataFrame(results), on=['date', 'fund_code'], validate='one_to_one')
    selected['exposure_group'] = selected['group']
    selected['eligible_research_sample'] = selected['group'].eq('AI Heavy') & selected.fit_pass
    selected.loc[selected['group'].eq('AI Heavy') & ~selected.fit_pass, 'group'] = 'Other'
    aggregate = aggregate.copy().set_index('date')
    for group, prefix in [('AI Heavy', 'heavy'), ('Control', 'control')]:
        subset = selected[selected.group == group].groupby('date')
        aggregate[f'{prefix}_fund_count'] = subset.fund_code.nunique().reindex(aggregate.index, fill_value=0)
        aggregate[f'{prefix}_median_corr'] = subset.corr_20d.median().reindex(aggregate.index)
        aggregate[f'{prefix}_positioning'] = -aggregate[f'{prefix}_median_corr']
    aggregate['diff_signal'] = aggregate.heavy_positioning - aggregate.control_positioning
    return selected, aggregate.reset_index()


def score_fit(basket, benchmark, train_days=40):
    x, y = np.asarray(basket, dtype=float), np.asarray(benchmark, dtype=float)
    if len(x) != len(y) or len(x) <= train_days:
        raise ValueError('Need aligned training and validation observations')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Missing observations cannot be fitted')
    if np.std(x[:train_days]) < 1e-12:
        return dict(alpha=np.nan, beta=np.nan, validation_corr=np.nan,
                    oos_r2=np.nan, validation_rmse=np.nan)
    beta, alpha = np.linalg.lstsq(np.column_stack([x[:train_days], np.ones(train_days)]), y[:train_days], rcond=None)[0]
    actual = y[train_days:]
    residual = actual - (alpha + beta * x[train_days:])
    # Benchmark forecast is the training mean, never the validation mean.
    denominator = np.sum((actual - y[:train_days].mean()) ** 2)
    corr = np.corrcoef(x[train_days:], actual)[0, 1] if np.std(x[train_days:]) > 1e-12 and np.std(actual) > 1e-12 else np.nan
    return dict(alpha=alpha, beta=beta, validation_corr=corr,
                oos_r2=1 - np.sum(residual ** 2) / denominator if denominator > 1e-20 else np.nan,
                validation_rmse=np.sqrt(np.mean(residual ** 2)))


def diagnose_holdings(holdings, stocks, aggregate, latest):
    asof = aggregate.date.max()
    benchmark = aggregate.set_index('date').ai_return.sort_index()
    prices = stocks.pivot(index='date', columns='stock_code', values='daily_return')
    rows = []
    for fund in latest.itertuples(index=False):
        positions = holdings[(holdings.fund_code == fund.fund_code) & (holdings.report_date <= asof)]
        report = positions.report_date.max()
        positions = positions[positions.report_date == report].nlargest(10, 'weight')
        total = positions.weight.sum()
        dates = benchmark.index[(benchmark.index > report) & (benchmark.index <= asof)][-c.FIT_WINDOW:]
        weights = positions.groupby('stock_code').weight.sum()
        returns = prices.reindex(index=dates, columns=weights.index)
        coverage = returns.notna().mul(weights, axis=1).sum(axis=1) / total if total > 0 else pd.Series(0., index=dates)
        row = dict(fund_code=fund.fund_code, fund_name=fund.fund_name, group=fund.group,
                   as_of=asof, report_date=report, reported_ai_exposure=fund.reported_ai_exposure,
                   top10_weight=total, observations=len(dates), min_quote_coverage=coverage.min(),
                   validation_start=dates[c.FIT_TRAIN_DAYS] if len(dates) > c.FIT_TRAIN_DAYS else pd.NaT,
                   alpha=np.nan, beta=np.nan, validation_corr=np.nan, oos_r2=np.nan,
                   validation_rmse=np.nan, fit_pass=False)
        if len(dates) < c.FIT_WINDOW:
            row['fit_status'] = '数据不足'
        elif total <= 0 or coverage.min() < c.FIT_MIN_COVERAGE:
            row['fit_status'] = '行情覆盖不足'
        else:
            # Normalization is only for the basket diagnostic; NAV-error weights remain original.
            basket = returns.mul(weights, axis=1).sum(axis=1) / total
            row.update(score_fit(basket, benchmark.reindex(dates), c.FIT_TRAIN_DAYS))
            row['fit_pass'] = bool(row['beta'] > 0 and row['validation_corr'] >= c.FIT_MIN_CORR and row['oos_r2'] >= c.FIT_MIN_OOS_R2)
            row['fit_status'] = '通过' if row['fit_pass'] else '未通过'
        rows.append(row)
    return pd.DataFrame(rows)
