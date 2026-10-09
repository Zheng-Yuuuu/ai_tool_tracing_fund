import unittest
import numpy as np
import pandas as pd
from plot_result import daily_error_comparison
from fit_hardware import score_fit, diagnose_holdings, select_by_historical_fit


class ExperimentTests(unittest.TestCase):
    def test_selection_ignores_current_and_future_prices(self):
        dates = pd.bdate_range('2025-01-02', periods=65)
        market = np.sin(np.arange(65)) * .01
        stocks = pd.DataFrame(dict(date=dates, stock_code='x', daily_return=market))
        holdings = pd.DataFrame(dict(fund_code=['a'], report_date=pd.to_datetime(['2024-12-31']), stock_code=['x'], weight=[.3]))
        aggregate = pd.DataFrame(dict(date=dates, ai_return=market))
        signals = pd.DataFrame(dict(date=dates, fund_code='a', group='AI Heavy', corr_20d=0.))
        result, _ = select_by_historical_fit(holdings, stocks, aggregate, signals)
        self.assertFalse(result.fit_pass.iloc[59])
        self.assertTrue(result.fit_pass.iloc[60])
        changed = stocks.copy()
        changed.loc[changed.date >= dates[60], 'daily_return'] = 100
        again, _ = select_by_historical_fit(holdings, changed, aggregate, signals)
        pd.testing.assert_frame_equal(result.iloc[:61], again.iloc[:61])
        self.assertEqual(result.fit_window_end.iloc[60], dates[59])

    def test_mean_before_correlation_and_empty_group(self):
        dates = pd.bdate_range('2025-01-01', periods=21)
        market = np.sin(np.arange(21))
        aggregate = pd.DataFrame(dict(date=dates, ai_return=market, ai_index=100., heavy_fund_count=2))
        errors = pd.concat([pd.DataFrame(dict(date=dates, fund_code=code, error=e)) for code, e in [('a', market), ('b', -3 * market)]])
        signals = errors[['date', 'fund_code']].assign(group='AI Heavy')
        result = daily_error_comparison(aggregate, errors, signals)
        self.assertAlmostEqual(result.heavy_error_corr_20d.iloc[-1], -1)
        self.assertTrue(result.heavy_error_corr_20d.iloc[:19].isna().all())
        signals.loc[signals.date == dates[-1], 'group'] = 'Other'
        result = daily_error_comparison(aggregate, errors, signals)
        self.assertTrue(pd.isna(result.heavy_error_corr_20d.iloc[-1]))

    def test_holdout_does_not_refit_and_negative_direction(self):
        x = np.sin(np.arange(60)) * .01
        good = score_fit(x, 2 * x + .001)
        self.assertAlmostEqual(good['oos_r2'], 1)
        self.assertAlmostEqual(good['beta'], 2)
        shifted = 2 * x + .001
        shifted[40:] *= -1
        bad = score_fit(x, shifted)
        self.assertAlmostEqual(bad['beta'], 2)
        self.assertLess(bad['oos_r2'], 0)

    def test_missing_holding_prices_rejected(self):
        dates = pd.bdate_range('2025-01-02', periods=60)
        holdings = pd.DataFrame(dict(fund_code=['a','a'], report_date=pd.to_datetime(['2024-12-31']*2), stock_code=['x','missing'], weight=[.4,.2]))
        stocks = pd.DataFrame(dict(date=dates, stock_code='x', daily_return=np.sin(np.arange(60))*.01))
        aggregate = pd.DataFrame(dict(date=dates, ai_return=stocks.daily_return))
        latest = pd.DataFrame(dict(fund_code=['a'], fund_name=['a'], group=['AI Heavy'], reported_ai_exposure=[.4]))
        result = diagnose_holdings(holdings, stocks, aggregate, latest)
        self.assertEqual(result.fit_status.iloc[0], '行情覆盖不足')
        self.assertFalse(result.fit_pass.iloc[0])


if __name__ == '__main__':
    unittest.main()
