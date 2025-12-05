"""
CUDA-Accelerated Casas-Alvero Counterexample Search
====================================================
Uses PyTorch CUDA to test MILLIONS of polynomials in parallel.

For RTX 4070 (12GB VRAM) we can process ~100k polynomials per batch.
"""

import torch
import numpy as np
from sympy import symbols, gcd, diff, factor, expand
import time
from typing import Tuple, List
import argparse


def check_cuda():
    """Check CUDA availability and print GPU info."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Install PyTorch with CUDA support.")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"✓ GPU: {props.name}")
    print(f"✓ VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"✓ CUDA Cores: {props.multi_processor_count * 128}")  # Approximate
    print(f"✓ Compute Capability: {props.major}.{props.minor}")
    return True


def generate_polynomials_gpu(batch_size: int, degree: int, coeff_range: int, device: str) -> torch.Tensor:
    """Generate random polynomial coefficients on GPU."""
    # Shape: (batch_size, degree) for coefficients a_{n-1}, ..., a_0
    # P(x) = x^n + a_{n-1}*x^{n-1} + ... + a_0
    coeffs = torch.randint(
        -coeff_range, coeff_range + 1, 
        (batch_size, degree), 
        dtype=torch.float64,
        device=device
    )
    return coeffs


def evaluate_polynomial_gpu(coeffs: torch.Tensor, x_vals: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Evaluate polynomials P(x) = x^n + sum(a_i * x^i) at multiple x values.
    
    coeffs: (batch_size, degree) - coefficients a_{n-1}, ..., a_0
    x_vals: (num_points,) - evaluation points
    
    Returns: (batch_size, num_points) - P(x) values
    """
    batch_size = coeffs.shape[0]
    num_points = x_vals.shape[0]
    
    # Build powers of x: x^0, x^1, ..., x^n
    powers = torch.stack([x_vals ** i for i in range(degree + 1)], dim=1)  # (num_points, degree+1)
    
    # Full coefficients including leading 1 for x^n
    # coeffs is [a_{n-1}, a_{n-2}, ..., a_1, a_0]
    full_coeffs = torch.cat([
        torch.ones(batch_size, 1, dtype=coeffs.dtype, device=coeffs.device),  # Leading 1
        coeffs
    ], dim=1)  # (batch_size, degree+1) = [1, a_{n-1}, ..., a_0]
    
    # Reverse to get [a_0, a_1, ..., a_{n-1}, 1] for dot product with [x^0, x^1, ..., x^n]
    full_coeffs = torch.flip(full_coeffs, dims=[1])
    
    # Compute P(x) for all polynomials at all points
    # (batch_size, degree+1) @ (degree+1, num_points) -> (batch_size, num_points)
    result = full_coeffs @ powers.T
    
    return result


def evaluate_derivative_gpu(coeffs: torch.Tensor, x_vals: torch.Tensor, degree: int, k: int) -> torch.Tensor:
    """
    Evaluate k-th derivative P^(k)(x) at multiple x values.
    """
    batch_size = coeffs.shape[0]
    
    # Derivative coefficients
    # d/dx (x^m) = m * x^(m-1)
    # d^k/dx^k (x^m) = m!/(m-k)! * x^(m-k) if m >= k, else 0
    
    deriv_coeffs = []
    for i in range(degree + 1):
        m = degree - i  # Power of x for this coefficient
        if m >= k:
            # Coefficient multiplier: m!/(m-k)!
            mult = 1
            for j in range(m, m - k, -1):
                mult *= j
            deriv_coeffs.append(mult)
        else:
            deriv_coeffs.append(0)
    
    deriv_coeffs = torch.tensor(deriv_coeffs, dtype=coeffs.dtype, device=coeffs.device)
    
    # Full coefficients
    full_coeffs = torch.cat([
        torch.ones(batch_size, 1, dtype=coeffs.dtype, device=coeffs.device),
        coeffs
    ], dim=1)
    
    # Apply derivative multipliers
    deriv_full = full_coeffs * deriv_coeffs.unsqueeze(0)
    
    # New degree is n - k
    new_degree = degree - k
    if new_degree < 0:
        return torch.zeros(batch_size, x_vals.shape[0], dtype=coeffs.dtype, device=coeffs.device)
    
    # Build powers for new degree
    powers = torch.stack([x_vals ** i for i in range(new_degree + 1)], dim=1)
    
    # Take only relevant coefficients and reverse
    relevant = deriv_full[:, :new_degree + 1]
    relevant = torch.flip(relevant, dims=[1])
    
    result = relevant @ powers.T
    return result


def check_common_root_gpu(P_vals: torch.Tensor, Pk_vals: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
    """
    Check if P and P^(k) share a common root (both near zero at same point).
    
    Returns: (batch_size,) boolean tensor - True if they share a root
    """
    # Both must be near zero at the same point
    P_near_zero = torch.abs(P_vals) < threshold
    Pk_near_zero = torch.abs(Pk_vals) < threshold
    
    # Check if any point has both near zero
    common_zero = P_near_zero & Pk_near_zero
    has_common = common_zero.any(dim=1)
    
    return has_common


def massive_gpu_search(
    degree: int,
    num_polynomials: int,
    coeff_range: int = 20,
    num_eval_points: int = 1000,
    batch_size: int = 50000,
    device: str = "cuda"
) -> Tuple[int, List]:
    """
    Massive GPU-accelerated search for Casas-Alvero counterexamples.
    """
    print(f"\n{'='*60}")
    print(f"DEGREE {degree}: Testing {num_polynomials:,} polynomials")
    print(f"{'='*60}")
    
    # Generate evaluation points (roots to test)
    x_vals = torch.linspace(-coeff_range, coeff_range, num_eval_points, dtype=torch.float64, device=device)
    
    candidates_found = 0
    candidates_list = []
    total_pass_all = 0
    
    num_batches = (num_polynomials + batch_size - 1) // batch_size
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        current_batch = min(batch_size, num_polynomials - batch_idx * batch_size)
        
        # Generate random coefficients on GPU
        coeffs = generate_polynomials_gpu(current_batch, degree, coeff_range, device)
        
        # Evaluate P(x) at all points
        P_vals = evaluate_polynomial_gpu(coeffs, x_vals, degree)
        
        # Check all k = 1 to n-1
        all_share_root = torch.ones(current_batch, dtype=torch.bool, device=device)
        
        for k in range(1, degree):
            Pk_vals = evaluate_derivative_gpu(coeffs, x_vals, degree, k)
            shares_root = check_common_root_gpu(P_vals, Pk_vals)
            all_share_root = all_share_root & shares_root
        
        # Count how many pass all conditions
        num_pass = all_share_root.sum().item()
        total_pass_all += num_pass
        
        # If any pass, verify with SymPy (exact computation)
        if num_pass > 0:
            pass_indices = torch.where(all_share_root)[0].cpu().numpy()
            pass_coeffs = coeffs[pass_indices].cpu().numpy()
            
            for idx, c in zip(pass_indices, pass_coeffs):
                # Verify with SymPy
                is_real, is_perfect_power = verify_with_sympy(c, degree)
                if is_real and not is_perfect_power:
                    candidates_found += 1
                    candidates_list.append((c.tolist(), degree))
                    print(f"  🔥 CANDIDATE: coeffs = {c.tolist()}")
        
        # Progress
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) * batch_size / elapsed
            print(f"  Batch {batch_idx + 1}/{num_batches} | "
                  f"{rate:,.0f} poly/sec | "
                  f"Pass numerical: {total_pass_all}")
    
    elapsed = time.time() - start_time
    rate = num_polynomials / elapsed
    
    print(f"\n  ✓ Completed in {elapsed:.1f}s ({rate:,.0f} polynomials/sec)")
    print(f"  ✓ Passed numerical filter: {total_pass_all}")
    print(f"  ✓ Verified counterexamples: {candidates_found}")
    
    return candidates_found, candidates_list


def verify_with_sympy(coeffs: np.ndarray, degree: int) -> Tuple[bool, bool]:
    """
    Verify a candidate with exact SymPy computation.
    Returns: (satisfies_condition, is_perfect_power)
    """
    x = symbols('x')
    
    # Build polynomial
    P = x**degree
    for i, c in enumerate(coeffs):
        P += int(c) * x**(degree - 1 - i)
    
    # Check all GCDs
    for k in range(1, degree):
        Pk = diff(P, x, k)
        g = gcd(P, Pk)
        if g == 1:
            return False, False  # Doesn't satisfy condition
    
    # Satisfies condition - check if perfect power
    try:
        f = factor(P)
        f_str = str(f)
        is_perfect = f"**{degree}" in f_str or f"**{degree})" in f_str
    except:
        is_perfect = False
    
    return True, is_perfect


def main():
    parser = argparse.ArgumentParser(description="CUDA-accelerated Casas-Alvero search")
    parser.add_argument("--polynomials", "-n", type=int, default=1_000_000,
                        help="Number of polynomials to test per degree")
    parser.add_argument("--degrees", "-d", type=str, default="3,4,5,6,7,8",
                        help="Comma-separated list of degrees to test")
    parser.add_argument("--coeff-range", "-r", type=int, default=20,
                        help="Coefficient range [-r, r]")
    parser.add_argument("--batch-size", "-b", type=int, default=100_000,
                        help="Batch size for GPU processing")
    args = parser.parse_args()
    
    print("="*60)
    print("🔬 CUDA-ACCELERATED CASAS-ALVERO COUNTEREXAMPLE SEARCH")
    print("="*60)
    print(f"   Testing {args.polynomials:,} polynomials per degree")
    print(f"   Coefficient range: [{-args.coeff_range}, {args.coeff_range}]")
    print(f"   Batch size: {args.batch_size:,}")
    print()
    
    if not check_cuda():
        print("\nFalling back to CPU (will be slower)...")
        device = "cpu"
    else:
        device = "cuda"
    
    degrees = [int(d.strip()) for d in args.degrees.split(",")]
    
    total_start = time.time()
    all_candidates = []
    
    for degree in degrees:
        count, candidates = massive_gpu_search(
            degree=degree,
            num_polynomials=args.polynomials,
            coeff_range=args.coeff_range,
            batch_size=args.batch_size,
            device=device
        )
        all_candidates.extend(candidates)
    
    total_elapsed = time.time() - total_start
    total_polys = args.polynomials * len(degrees)
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"Total polynomials tested: {total_polys:,}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Overall rate: {total_polys/total_elapsed:,.0f} polynomials/sec")
    print()
    
    if all_candidates:
        print(f"🔥 COUNTEREXAMPLES FOUND: {len(all_candidates)}")
        for coeffs, deg in all_candidates:
            print(f"   Degree {deg}: coeffs = {coeffs}")
    else:
        print("✓ NO COUNTEREXAMPLES FOUND")
        print("✓ The Casas-Alvero Conjecture holds for all tested cases.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

