import unittest

from statistical_analysis import holm_adjust, mean_confidence_interval, paired_significance


class StatisticalAnalysisTest(unittest.TestCase):
    def test_confidence_interval(self):
        result = mean_confidence_interval([1, 2, 3, 4, 5])
        self.assertEqual(result["n"], 5)
        self.assertEqual(result["mean"], 3.0)
        self.assertLess(result["ci_low"], result["mean"])
        self.assertGreater(result["ci_high"], result["mean"])

    def test_paired_significance(self):
        result = paired_significance([2, 3, 4, 5, 6], [1, 2, 3, 4, 4])
        self.assertGreater(result["mean_difference"], 0)
        self.assertIn("paired_t_pvalue", result)
        self.assertIn("cohen_dz", result)

    def test_holm_adjustment(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(len(adjusted), 3)
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))


if __name__ == "__main__":
    unittest.main()
