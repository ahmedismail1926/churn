"""
Unit tests for resampling module
"""
import unittest
import pandas as pd
import numpy as np
from collections import Counter
from resampling import (
    print_class_distribution,
    resample_data_smoteenn,
    resample_data_smote,
    resample_data_smotetomek,
    resample_data_adasyn,
    apply_best_resampling
)


class TestResampling(unittest.TestCase):
    """Test cases for resampling module"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests"""
        # Create a small imbalanced dataset for testing
        np.random.seed(42)
        
        # Majority class (500 samples)
        X_majority = np.random.randn(500, 10)
        y_majority = np.zeros(500)
        
        # Minority class (100 samples)
        X_minority = np.random.randn(100, 10)
        y_minority = np.ones(100)
        
        # Combine
        cls.X = pd.DataFrame(np.vstack([X_majority, X_minority]))
        cls.y = pd.Series(np.concatenate([y_majority, y_minority]))
        
        print(f"\nTest data created: {cls.X.shape[0]} samples, {cls.X.shape[1]} features")
        print(f"Class distribution: {Counter(cls.y)}")
    
    def test_print_class_distribution(self):
        """Test class distribution printing function"""
        print("\n" + "="*60)
        print("TEST: print_class_distribution")
        print("="*60)
        
        counts = print_class_distribution(self.y, "Test Dataset")
        
        self.assertIsInstance(counts, dict)
        self.assertEqual(len(counts), 2)
        self.assertEqual(counts[0], 500)
        self.assertEqual(counts[1], 100)
        
        print("[OK] Class distribution printing works correctly")
    
    def test_smoteenn_resampling(self):
        """Test SMOTE-ENN resampling"""
        print("\n" + "="*60)
        print("TEST: SMOTE-ENN Resampling")
        print("="*60)
        
        X_res, y_res, stats = resample_data_smoteenn(
            self.X, self.y,
            random_state=42,
            verbose=True
        )
        
        # Check output types
        self.assertIsInstance(X_res, pd.DataFrame)
        self.assertIsInstance(y_res, pd.Series)
        self.assertIsInstance(stats, dict)
        
        # Check stats content
        self.assertIn('method', stats)
        self.assertIn('runtime', stats)
        self.assertIn('original_samples', stats)
        self.assertIn('resampled_samples', stats)
        
        # Check that resampling was performed
        self.assertNotEqual(len(y_res), len(self.y))
        
        # Check that features are preserved
        self.assertEqual(X_res.shape[1], self.X.shape[1])
        
        print("[OK] SMOTE-ENN resampling works correctly")
    
    def test_smote_resampling(self):
        """Test SMOTE resampling"""
        print("\n" + "="*60)
        print("TEST: SMOTE Resampling")
        print("="*60)
        
        X_res, y_res, stats = resample_data_smote(
            self.X, self.y,
            random_state=42,
            verbose=False
        )
        
        # Check that classes are balanced
        counts = Counter(y_res)
        self.assertEqual(counts[0], counts[1])
        
        # Check that we have more samples than before
        self.assertGreater(len(y_res), len(self.y))
        
        print(f"Original samples: {len(self.y)}")
        print(f"Resampled samples: {len(y_res)}")
        print(f"Class balance: {counts}")
        print("[OK] SMOTE resampling works correctly")
    
    def test_smotetomek_resampling(self):
        """Test SMOTE-Tomek resampling"""
        print("\n" + "="*60)
        print("TEST: SMOTE-Tomek Resampling")
        print("="*60)
        
        X_res, y_res, stats = resample_data_smotetomek(
            self.X, self.y,
            random_state=42,
            verbose=False
        )
        
        # Check basic properties
        self.assertGreater(len(y_res), len(self.y))
        self.assertEqual(stats['method'], 'SMOTE-Tomek')
        
        print(f"Original samples: {len(self.y)}")
        print(f"Resampled samples: {len(y_res)}")
        print("[OK] SMOTE-Tomek resampling works correctly")
    
    def test_adasyn_resampling(self):
        """Test ADASYN resampling"""
        print("\n" + "="*60)
        print("TEST: ADASYN Resampling")
        print("="*60)
        
        X_res, y_res, stats = resample_data_adasyn(
            self.X, self.y,
            random_state=42,
            verbose=False
        )
        
        # Check basic properties
        self.assertGreater(len(y_res), len(self.y))
        self.assertEqual(stats['method'], 'ADASYN')
        
        print(f"Original samples: {len(self.y)}")
        print(f"Resampled samples: {len(y_res)}")
        print("[OK] ADASYN resampling works correctly")
    
    def test_apply_best_resampling(self):
        """Test wrapper function for applying best resampling"""
        print("\n" + "="*60)
        print("TEST: apply_best_resampling wrapper")
        print("="*60)
        
        # Test with different methods
        methods = ['smoteenn', 'smote', 'smotetomek', 'adasyn']
        
        for method in methods:
            print(f"\nTesting method: {method}")
            X_res, y_res, stats = apply_best_resampling(
                self.X, self.y,
                method=method,
                random_state=42,
                verbose=False
            )
            
            self.assertIsInstance(X_res, pd.DataFrame)
            self.assertIsInstance(y_res, pd.Series)
            self.assertEqual(stats['method'].lower(), method.replace('-', ''))
            print(f"  [OK] {method} works via wrapper")
        
        print("\n[OK] Wrapper function works for all methods")
    
    def test_invalid_method(self):
        """Test that invalid method raises error"""
        print("\n" + "="*60)
        print("TEST: Invalid method handling")
        print("="*60)
        
        with self.assertRaises(ValueError):
            apply_best_resampling(self.X, self.y, method='invalid_method')
        
        print("[OK] Invalid method correctly raises ValueError")
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed"""
        print("\n" + "="*60)
        print("TEST: Reproducibility")
        print("="*60)
        
        # Run SMOTE-ENN twice with same seed
        X_res1, y_res1, _ = resample_data_smoteenn(
            self.X, self.y,
            random_state=42,
            verbose=False
        )
        
        X_res2, y_res2, _ = resample_data_smoteenn(
            self.X, self.y,
            random_state=42,
            verbose=False
        )
        
        # Check that results are identical
        pd.testing.assert_frame_equal(X_res1, X_res2)
        pd.testing.assert_series_equal(y_res1, y_res2)
        
        print("[OK] Results are reproducible with same random seed")


def run_tests():
    """Run all tests with detailed output"""
    print("\n" + "="*70)
    print("RESAMPLING MODULE - UNIT TESTS")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestResampling)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"\nTests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[OK] ALL TESTS PASSED!")
    else:
        print("\n[FAILED] Some tests failed!")
    
    print("="*70)
    
    return result


if __name__ == "__main__":
    run_tests()
