import numpy as np
from scipy.special import elliprj, elliprf
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def jacobi_test_case(b_over_a_target=0.9, debug=False):
    """
    Generate a valid Jacobi ellipsoid configuration.
    
    Parameters
    ----------
    b_over_a_target : float
        Target ratio b/a (will solve for exact value that satisfies constraints)
        Should be in range (0, 1) for oblate-ish shapes
    debug : bool
        If True, plot the residual function
        
    Returns
    -------
    dict with keys:
        'a', 'b', 'c' : semi-axes (a >= b >= c)
        'omega_squared' : rotation rate squared
        'b_over_a' : actual b/a ratio found
        'c_over_a' : actual c/a ratio
    """
    
    # Fix a = 1 for simplicity
    a = 1.0
    
    def jacobi_residual(b_over_a):
        """Residual function for Jacobi condition with abc=1 constraint"""
        b = a * b_over_a
        c = 1.0 / (a * b)  # From abc = 1
        
        a2, b2, c2 = a**2, b**2, c**2
        
        
        print(f"a, b, c: {a, b, c}")
        print(f"1/c^2: {1/c2}; (1/a^2+1/b^2): {1/a2+1/b2}")
        
        
        # Check ordering constraint
        if not (a >= b >= c > 0):
            return np.inf
        
        # When b = a, we have division by zero, so avoid it
        if np.abs(b - a) < 1e-10:
            return np.inf
        
        # Compute R_J elliptic integrals
        
        
        print(f"a, b, c: {a, b, c}")
        print(f"1/c^2: {1/c2}; (1/a^2+1/b^2): {1/a2+1/b2}")
        
        try:
            RJ_a = elliprj(b2, c2, a2, a2)
            RJ_b = elliprj(a2, c2, b2, b2)
            RJ_c = elliprj(a2, b2, c2, c2)
        except:
            return np.inf
        
        # Jacobi condition
        LHS = (a2 * b2) / (b2 - a2) * (RJ_a - RJ_b)
        RHS = c2 * RJ_c
        
        return LHS - RHS
    
    # Debug: plot the residual
    if debug:
        b_vals = np.linspace(0.1, 0.99, 200)
        residuals = [jacobi_residual(b) for b in b_vals]
        
        plt.figure(figsize=(10, 6))
        plt.plot(b_vals, residuals)
        plt.axhline(y=0, color='r', linestyle='--', label='Zero')
        plt.xlabel('b/a')
        plt.ylabel('Jacobi Residual')
        plt.title('Jacobi Condition Residual vs b/a ratio')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        print(f"Residual at b/a=0.1: {jacobi_residual(0.1)}")
        print(f"Residual at b/a=0.5: {jacobi_residual(0.5)}")
        print(f"Residual at b/a=0.9: {jacobi_residual(0.9)}")
    
    # Find the b/a ratio that satisfies the Jacobi condition
    # Search in a reasonable range
    b_over_a = brentq(jacobi_residual, 0.1, 0.99)
    
    # Compute final configuration
    b = a * b_over_a
    c = 1.0 / (a * b)
    
    # Compute omega^2 from virial theorem
    a2, b2, c2 = a**2, b**2, c**2
    
    # A_i integrals
    RF = elliprf(b2, c2, a2)
    RJ_a = elliprj(b2, c2, a2, a2)
    A1 = 2 * (RF - RJ_a / 3)
    
    RF = elliprf(a2, b2, c2)
    RJ_c = elliprj(a2, b2, c2, c2)
    A3 = 2 * (RF - RJ_c / 3)
    
    omega_squared = 2 * np.pi * (A1 - A3)
    
    return {
        'a': a,
        'b': b,
        'c': c,
        'omega_squared': omega_squared,
        'b_over_a': b_over_a,
        'c_over_a': c / a,
        'abc': a * b * c
    }


# Test it with debug mode first
if __name__ == "__main__":
    config = jacobi_test_case(0.9, debug=True)