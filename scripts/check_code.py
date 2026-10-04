"""Run contract and in-memory physics checks without training or rendering."""
from pathlib import Path
import sys
import unittest


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "tests"))
loader = unittest.TestLoader()
suite = unittest.TestSuite()
for module in ("test_contracts", "test_primitives", "test_physics"):
    suite.addTests(loader.loadTestsFromName(module))
from test_core import CoreTests
from test_integration import IntegrationTests, VisionContractTests, TrackerContractTests
from test_audit import AuditCoreTests
from test_release import ReportTests
for test_class, excluded in ((CoreTests, {"test_family_split_and_future_isolation"}),
                             (IntegrationTests, {"test_evaluation_preserves_family_units"}),
                             (VisionContractTests, set()), (TrackerContractTests, set()),
                             (AuditCoreTests, set()), (ReportTests, set())):
    for name in loader.getTestCaseNames(test_class):
        if name not in excluded:
            suite.addTest(test_class(name))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
