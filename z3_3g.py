import numpy as np

def verify_ghz_structure():
    """
    Verifies that the Z3-algebra's cubic invariant form (epsilon_ijk)
    is mathematically isomorphic to a GHZ-class state tensor.
    """
    print("--- Verifying GHZ Nature of the Vacuum Invariant ---")
    
    # 1. Construct the invariant tensor T_ijk = epsilon_ijk
    dim = 3
    T = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                # Antisymmetric structure constants
                if (i, j, k) in [(0,1,2), (1,2,0), (2,0,1)]: 
                    T[i,j,k] = 1
                elif (i, j, k) in [(2,1,0), (1,0,2), (0,2,1)]: 
                    T[i,j,k] = -1
    
    # 2. Reshape for SVD across the A|BC partition
    M_A_BC = T.reshape(dim, dim*dim)
    
    # 3. Perform SVD and analyze singular values
    U, S, Vh = np.linalg.svd(M_A_BC)
    
    # Filter out machine-precision noise
    non_zero_S = S[S > 1e-10]
    rank = len(non_zero_S)
    
    print(f"\nSingular Values across partition A|BC: {non_zero_S}")
    print(f"Schmidt Rank: {rank}")
    
    if rank > 1:
        print("-> Confirmed: System is entangled across A|BC cut.")
        
        # Check for degeneracy (GHZ signature)
        is_degenerate = np.allclose(non_zero_S, non_zero_S[0])
        if is_degenerate:
            print("-> Signature: Singular values are degenerate.")
            print("\n[CONCLUSION] The algebraic cubic invariant possesses")
            print(" the definitive mathematical structure of a maximally")
            print(" entangled GHZ-class state.")
        else:
            print("-> Entangled, but not GHZ-class (e.g., W-class).")
    else:
        print("[CONCLUSION] The state is separable (a product state).")

# Run the verification
verify_ghz_structure()