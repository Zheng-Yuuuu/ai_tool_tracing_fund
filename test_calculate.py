import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from calculate import calculate


class CalculationTests(unittest.TestCase):
    def test_formula_and_sign(self):
        dates = pd.bdate_range('2025-01-01', periods=40)
        returns = np.sin(np.arange(40)) * .02
        pool = pd.DataFrame({'stock_code': ['300308']})
        stocks = pd.DataFrame({'date': dates, 'stock_code': '300308', 'daily_return': returns})
        holdings = pd.DataFrame({'fund_code': ['000001', '000002'], 'report_date': pd.to_datetime(['2024-12-31'] * 2), 'stock_code': ['300308'] * 2, 'weight': [.6, .04]})
        nav = pd.concat([pd.DataFrame({'date': dates, 'fund_code': code, 'fund_name': code, 'daily_return': factor * returns}) for code, factor in [('000001', .8), ('000002', .04)]])
        errors, signals, aggregate = calculate(nav, holdings, stocks, pool)
        heavy = errors[errors.fund_code == '000001']
        np.testing.assert_allclose(heavy.estimated_return, .6 * returns)
        np.testing.assert_allclose(heavy.error, -.2 * returns)
        self.assertTrue((heavy.covered_weight == .6).all())
        self.assertAlmostEqual(signals[signals.fund_code == '000001'].corr_20d.iloc[-1], -1)
        self.assertAlmostEqual(aggregate.heavy_positioning.iloc[-1], 1)
        self.assertTrue(aggregate.control_positioning.isna().all())
        self.assertTrue(aggregate.diff_signal.isna().all())

    def test_missing_and_report_switch(self):
        dates = pd.to_datetime(['2025-01-02', '2025-04-01'])
        nav = pd.DataFrame({'date': dates, 'fund_code': '000001', 'fund_name': 'x', 'daily_return': [.01, .01]})
        holdings = pd.DataFrame({'fund_code': ['000001'] * 2, 'report_date': pd.to_datetime(['2024-12-31', '2025-03-31']), 'stock_code': ['300308'] * 2, 'weight': [.2, .3]})
        stocks = pd.DataFrame({'date': dates, 'stock_code': '300308', 'daily_return': [np.nan, .02]})
        errors, _, _ = calculate(nav, holdings, stocks, pd.DataFrame({'stock_code': ['300308']}))
        self.assertTrue(pd.isna(errors.estimated_return.iloc[0]))
        self.assertEqual(errors.covered_weight.iloc[0], 0)
        self.assertAlmostEqual(errors.estimated_return.iloc[1], .006)
        self.assertAlmostEqual(errors.error.iloc[1], -.004)

    def test_diff_signal_and_no_future_report(self):
        dates = pd.bdate_range('2025-01-01', periods=35)
        returns = np.cos(np.arange(35)) * .02
        stocks = pd.DataFrame({'date': dates, 'stock_code': '300308', 'daily_return': returns})
        holdings = pd.DataFrame({'fund_code': ['000001', '000002', '000003'], 'report_date': pd.to_datetime(['2024-12-31', '2024-12-31', '2025-03-31']), 'stock_code': '300308', 'weight': [.6, .04, .3]})
        nav = pd.concat([pd.DataFrame({'date': dates, 'fund_code': code, 'fund_name': code, 'daily_return': factor * returns}) for code, factor in [('000001', .8), ('000002', 0), ('000003', .5)]])
        errors, _, aggregate = calculate(nav, holdings, stocks, pd.DataFrame({'stock_code': ['300308']}))
        self.assertAlmostEqual(aggregate.diff_signal.iloc[-1], 2)
        self.assertTrue(errors[errors.fund_code == '000003'].estimated_return.isna().all())
        self.assertTrue(aggregate.diff_signal.iloc[:19].isna().all())

    @patch('config.AI_HEAVY_THRESHOLD', .40)
    def test_strict_40_percent_and_separate_exposure_pool(self):
        date = pd.Timestamp('2025-01-02')
        nav = pd.DataFrame({'date': [date] * 2, 'fund_code': ['000001', '000002'], 'fund_name': ['x', 'y'], 'daily_return': [0, 0]})
        holdings = pd.DataFrame({'fund_code': ['000001'] * 3 + ['000002'], 'report_date': [pd.Timestamp('2024-12-31')] * 4, 'stock_code': ['300308', '300502', '002463', '300308'], 'weight': [.1, .2, .1, .4001]})
        stocks = pd.DataFrame({'date': [date] * 3, 'stock_code': ['300308', '300502', '002463'], 'daily_return': [.01, .02, .03]})
        index_pool = pd.DataFrame({'stock_code': ['300308']})
        exposure_pool = pd.DataFrame({'stock_code': ['300308', '300502', '002463']})
        _, signals, aggregate = calculate(nav, holdings, stocks, index_pool, exposure_pool)
        groups = signals.set_index('fund_code').group
        self.assertEqual(groups['000001'], 'Other')
        self.assertEqual(groups['000002'], 'AI Heavy')
        self.assertAlmostEqual(aggregate.ai_return.iloc[0], .01)


if __name__ == '__main__':
    unittest.main()
