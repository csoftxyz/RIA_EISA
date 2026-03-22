import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

# Configure aesthetic style
plt.style.use('default')

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-9: 
        return None
    return v / norm

def generate_44_lattice():
    # 1. Basic Seeds
    basis = np.eye(3)
    dem = np.array([1, 1, 1]) / np.sqrt(3)
    vectors = [basis[0], basis[1], basis[2], dem, -dem]
    
    # Triality Matrix (for reference)
    T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    
    # Roots (Type: [1, -1, 0])
    roots = []
    for i in range(3):
        for j in range(3):
            if i != j:
                r = np.zeros(3)
                r[i] = 1
                r[j] = -1
                roots.append(normalize(r))
    
    # Hybrids (Type: [2, -1, -1])
    hybrids = []
    for i in range(3):
        h = -np.ones(3)
        h[i] = 2
        hybrids.append(normalize(h))
        hybrids.append(normalize(-h))

    # Collect all unique vectors
    all_vecs = []
    all_vecs.extend(vectors)
    all_vecs.extend(roots)
    all_vecs.extend(hybrids)
    
    # Remove duplicates and Nones
    unique_vecs = []
    seen = set()
    for v in all_vecs:
        if v is not None:
            t = tuple(np.round(v, 6))
            if t not in seen:
                seen.add(t)
                unique_vecs.append(v)
                
    return np.array(unique_vecs)

def plot_z3_lattice():
    vectors = generate_44_lattice()
    
    fig = plt.figure(figsize=(16, 8))
    
    # --- Subplot 1: 3D "Luban Lock" Structure ---
    ax1 = fig.add_subplot(121, projection='3d')
    # 使用 raw string 修复 SyntaxWarning
    ax1.set_title(r"(a) The 44-Vector Vacuum Lattice ($\mathcal{L}_{44}$)", 
                  fontsize=14, y=1.05)
    
    # Plot vectors
    dem_axis = np.array([1, 1, 1]) / np.sqrt(3)
    
    for v in vectors:
        cos_theta = np.dot(v, dem_axis)
        
        if np.isclose(abs(cos_theta), 1.0):
            c, s, lw, z = 'crimson', 50, 2.5, 10   # Democratic (Axis)
        elif np.isclose(cos_theta, 0.0):
            c, s, lw, z = 'royalblue', 40, 2.0, 8  # Roots (Planar)
        else:
            c, s, lw, z = 'gray', 20, 1.0, 5       # Hybrids
            
        ax1.plot([0, v[0]], [0, v[1]], [0, v[2]], 
                 color=c, alpha=0.7, linewidth=lw)
        ax1.scatter(v[0], v[1], v[2], color=c, s=s, 
                    depthshade=False, zorder=z)

    # Wireframe sphere for context
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = 0.98 * np.cos(u) * np.sin(v)
    y = 0.98 * np.sin(u) * np.sin(v)
    z = 0.98 * np.cos(v)
    ax1.plot_wireframe(x, y, z, color="gainsboro", alpha=0.2)

    ax1.view_init(elev=20, azim=45)
    ax1.set_axis_off()
    ax1.text2D(0.05, 0.05, "Z$_3$ Triality Symmetry\n3D Isometric View", 
               transform=ax1.transAxes)

    # --- Subplot 2: 2D Projection (A2 Root System) ---
    ax2 = fig.add_subplot(122)
    ax2.set_title(r"(b) Projection onto $A_2$ Plane (Hexagonal)", 
                  fontsize=14, y=1.05)
    
    # Projection onto plane perpendicular to (1,1,1)
    u_vec = np.array([1, -1, 0]) / np.sqrt(2)
    v_vec = np.array([1, 1, -2]) / np.sqrt(6)
    
    for v in vectors:
        x_p = np.dot(v, u_vec)
        y_p = np.dot(v, v_vec)
        
        cos_theta = np.dot(v, dem_axis)
        if np.isclose(abs(cos_theta), 1.0):
            c, alpha = 'crimson', 0.2
        elif np.isclose(cos_theta, 0.0):
            c, alpha = 'royalblue', 1.0
        else:
            c, alpha = 'gray', 0.6
            
        ax2.plot([0, x_p], [0, y_p], color=c, alpha=alpha, linewidth=1.5)
        ax2.scatter(x_p, y_p, color=c, s=40, alpha=alpha, zorder=10)

    # Highlight the Hexagon
    circle = plt.Circle((0, 0), 1.0, color='royalblue', fill=False, 
                        linestyle='--', alpha=0.3)
    ax2.add_artist(circle)
    
    ax2.set_aspect('equal')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.axis('off')
    
    ax2.text(0, -1.3, "Perfect Hexagonal Symmetry ($C_6$)\n"
                      "Matches hBN Lattice Geometry", 
             ha='center', fontsize=12, color='darkblue')

    plt.tight_layout()
    plt.savefig('Fig_Z3_Lattice_Architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Figure generated: Fig_Z3_Lattice_Architecture.png")

if __name__ == "__main__":
    plot_z3_lattice()