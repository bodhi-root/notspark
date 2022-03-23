import unittest
import notspark.expressions.functions as F
from notspark.expressions.core import DefaultContext


class TestFunctions(unittest.TestCase):

    def test_lit(self):
        self.assertEqual(F.lit(1).eval(None), 1)

    def test_var(self):
        ctx = DefaultContext({"a": 2, "b": 3})
        self.assertEqual(F.var("a").eval(ctx), 2)
        self.assertEqual((F.var("a") + F.var("b")).eval(ctx), 5)
        self.assertEqual((F.var("a") * F.var("b")).eval(ctx), 6)
        self.assertEqual((F.var("a") + F.var("b") * 2).eval(ctx), 8)

    def test_other(self):
        x = F.lit([1, 2, 3])
        self.assertEqual((F.sum(x)).eval(None), 6)

        self.assertAlmostEqual((
            F.atan2(F.lit(172), F.lit(77))
        ).eval(None), 1.1498780382155047)

        self.assertAlmostEqual((F.corr([1,2,3], [4,5,7])).eval(None), 0.9819805060619659)


if __name__ == "__main__":
    unittest.main()
