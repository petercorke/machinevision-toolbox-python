import numpy.testing as nt
import unittest

from color import *

class TestColor(unittest.TestCase):

    def test_blackbody(self):
        e = blackbody(500e-9, 4000)
        self.assertAlmostEqual(e, 2.86512308e+12, delta=1e4)

        e = blackbody([400e-9, 500e-9], 4000)
        self.assertEqual(len(e), 2)
        self.assertAlmostEqual(e[0], 1.44662486e+12, delta=1e4)
        self.assertAlmostEqual(e[1], 2.86512308e+12, delta=1e4)

# ---------------------------------------------------------------------------------------#
if __name__ == '__main__':

    unittest.main()