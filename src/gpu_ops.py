import torch

def matrix_multiply_gpu(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication on GPU if available.
    
    Args:
        matrix1 (torch.Tensor): First input matrix
        matrix2 (torch.Tensor): Second input matrix
    
    Returns:
        torch.Tensor: Result of matrix multiplication
    
    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system")
    
    # Move matrices to GPU
    matrix1_gpu = matrix1.cuda()
    matrix2_gpu = matrix2.cuda()
    
    # Perform multiplication
    result_gpu = torch.matmul(matrix1_gpu, matrix2_gpu)
    
    # Move result back to CPU
    return result_gpu.cpu() 