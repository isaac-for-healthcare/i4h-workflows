import unittest
import coverage
import sys
import os

def run_tests_with_coverage():
    """Run all unittest cases with coverage reporting"""
    try:
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Add project root to Python path
        sys.path.insert(0, project_root)
        
        # Initialize coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Discover and run tests
        loader = unittest.TestLoader()
        tests_dir = os.path.join(project_root, 'tests')
        suite = loader.discover(tests_dir)
        
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