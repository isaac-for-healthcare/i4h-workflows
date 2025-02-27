import os
import sys
import traceback
import unittest

import coverage

PROJECT_ROOTS = [
    "workflows/robotic_ultrasound",
]


def run_tests_with_coverage(project_root):
    """Run all unittest cases with coverage reporting"""
    sys_path_restore = sys.path
    try:
        # Get the project root directory
        # Initialize coverage
        sys.path.append(os.path.join(project_root, 'scripts'))
        cov = coverage.Coverage()
        cov.start()

        # Discover and run tests

        suite = unittest.TestSuite()
        tests_dir = os.path.join(project_root, 'tests')
        for name in os.listdir(tests_dir):
            path = os.path.join(tests_dir, name)
            print(path)
            if os.path.isdir(path):
                loader = unittest.TestLoader()
                suite.addTest(loader.discover(path))


        # Run tests
        print("Running tests...")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # Stop coverage
        cov.stop()
        cov.save()

        # Print coverage report
        print("\nCoverage Report:")
        cov.report()

        # Generate HTML report
        cov.html_report(directory=os.path.join(project_root, 'htmlcov'))
        print("\nDetailed HTML coverage report generated in 'htmlcov' directory")

        # Return appropriate exit code
        if result.wasSuccessful():
            print("\nAll tests passed!")
            return 0
        else:
            print("\nSome tests failed!")
            return 1

    except Exception as e:
        print(f"Error running tests: {e}")
        print(traceback.format_exc())
        # restore sys.path
        sys.path = sys_path_restore
        return 1

if __name__ == "__main__":
    for project_root in PROJECT_ROOTS:
        sys.exit(run_tests_with_coverage(project_root))
