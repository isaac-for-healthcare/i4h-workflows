import unittest
import torch
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gpu_ops import matrix_multiply_gpu

class TestGPUOps(unittest.TestCase):
    def setUp(self):
        """Set up test matrices"""
        self.matrix1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.matrix2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        self.expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_matrix_multiply_gpu(self):
        """Test matrix multiplication on GPU"""
        result = matrix_multiply_gpu(self.matrix1, self.matrix2)
        self.assertTrue(torch.allclose(result, self.expected))
    
    @unittest.skipIf(torch.cuda.is_available(), "CUDA is available")
    def test_raises_error_when_no_cuda(self):
        """Test error raising when CUDA is not available"""
        with self.assertRaises(RuntimeError):
            matrix_multiply_gpu(self.matrix1, self.matrix2)

    def test_input_dimensions(self):
        """Test error handling for incompatible dimensions"""
        invalid_matrix = torch.tensor([[1.0, 2.0]])
        with self.assertRaises(RuntimeError):
            matrix_multiply_gpu(self.matrix1, invalid_matrix)

if __name__ == '__main__':
    unittest.main() 