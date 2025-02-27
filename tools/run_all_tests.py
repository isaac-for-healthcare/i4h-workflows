import os
import sys
import unittest

import coverage

libs = [
    'robotic_ultrasound',
]


def run_tests_with_coverage():
    """Run all unittest cases with coverage reporting"""
    try:
        # Get the project root directory
        import robotic_ultrasound
        project_root = robotic_ultrasound.__basedir__

        # Initialize coverage
        cov = coverage.Coverage()
        cov.start()

        # Discover and run tests
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        tests_dir = os.path.join(project_root, 'tests')
        for name in os.listdir(tests_dir):
            path = os.path.join(tests_dir, name)
            if os.path.isdir(path):
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
        return 1

if __name__ == "__main__":
    sys.exit(run_tests_with_coverage())
