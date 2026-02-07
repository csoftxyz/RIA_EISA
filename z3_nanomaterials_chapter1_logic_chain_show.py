# File name: z3_nanomaterials_chapter1_mindmap_vertical.py
# Description: Program to generate a vertical mindmap (Graphviz) for the logic chain in Chapter 1 of the Nanomaterials paper revision.
# Layout: Top to bottom (vertical flow) for clear process and easy LaTeX inclusion.
# No section numbers (1.1, 1.2 etc.) – focus on core formulas and reviewer concerns.
# Key nodes with different colors:
# - Central (yellow): Overall framework
# - Algebra/Uniqueness (blue): Core math (C = ε, uniqueness from Jacobi/Schur)
# - Coupling (lightblue): Linear current coupling
# - Renormalization (cyan): Pi(0) negative, M_eff^2
# - Naturalness (green): Beta=0 from triality
# - Condensate/Pairing (orange): V_eff, V_vac, Tc(d)
# - Validation (lightgreen): Quantitative (focus THz skin depth)
# - Discriminating (pink): Testable signatures
# - Constraints (red): Consistency/evasion
# Requirements: graphviz (pip install graphviz + system Graphviz dot executable)
# Run to generate 'z3_chapter1_mindmap_vertical.pdf'

import graphviz

print("=== Generating Vertical Mindmap for Nanomaterials Paper Chapter 1 Logic Chain ===\n")

# Create Digraph for mindmap
dot = graphviz.Digraph('Z3_Chapter1_Logic_Chain', format='pdf')
dot.attr(rankdir='TB', nodesep='0.8', ranksep='1.2')  # Vertical layout (top to bottom)
dot.attr('node', shape='box', style='filled', fontsize='13', fontname='Helvetica')
dot.attr('edge', arrowhead='vee', arrowsize='1.2')

# Central node (yellow)
dot.node('Central', 'Z3 Vacuum Inertia Mechanism\nExploratory Framework\n(Algebraic Origin of Mesoscopic Anomalies)', fillcolor='yellow', shape='ellipse')

# Anomalies & Proposal (lightblue)
dot.node('Anomalies', 'Anomalous THz Skin Depth Saturation\nTc Enhancement in Nanowires\nChallenge Conventional Models\nPropose Complementary Vacuum Inertia', fillcolor='lightblue')

# Algebra & Uniqueness (blue - core math)
dot.node('Algebra', 'Z3-Graded Lie Superalgebra\nGrading: g0⊕g1⊕g2 (12+4+3)\nBrackets + Cubic {F,F,ζ} = -C B_a\nC = ε_kαβ (Levi-Civita)\nUniqueness: Jacobi + Invariance + Schur\nProves: Unique Ternary Vacuum Sector', fillcolor='blue')

# Coupling (cyan)
dot.node('Coupling', 'Superconnection → dim-5 Operator\nQuasistatic: -g̃ J·A ζ\nProves: Scalar Channel Dominance\n(Parameter-Free Interaction)', fillcolor='cyan')

# Renormalization & Softening (lightcyan)
dot.node('Renorm', 'In-Medium Dyson Equation\nPi(0) < 0 (Attractive Channel)\nM_eff^2 = M_vac^2 - μ_med^2\nξ_vac ~ 50--100 nm\nProves: TeV-to-meV Softening', fillcolor='lightcyan')

# Naturalness (green)
dot.node('Naturalness', 'One-Loop Beta Function\nβ_m2 = 0 (Triality Trace Cancellation)\nProves: Protected Hierarchy\n(No Tuning Needed)', fillcolor='lightgreen')

# Condensate & Pairing (orange)
dot.node('Condensate', 'Condensate when M_eff^2 < 0\nV_eff = 1/2 M_eff^2 ζ^2 + λ ε ζ^3\nDemocratic VEV\nV_vac ~ -g̃^2 / M_eff^2\nTc(d) Modified McMillan (Exponential Enhancement)\nProves: Geometric Vacuum Pairing', fillcolor='orange')

# Quantitative Validation (lightgreen)
dot.node('Validation', 'Reproducible Validation\nTables + Theory-Experiment Overlays\nFocus: THz Skin Depth Plateau\nProves: Consistency with Data (Ab Initio)', fillcolor='palegreen')

# Discriminating Predictions (pink)
dot.node('Discrim', 'Discriminating Signatures\nPlateau ∝ 1/τ_vac\nNon-Monotonic R_s (High-RRR Only)\nIsotope-Independent Tc Enhancement\nProves: Distinguishable from Conventional', fillcolor='lightpink')

# Constraints (red)
dot.node('Constraints', 'Consistency with Constraints\nIn-Medium Screening + Triality Protection\nEvades Precision QED/Photon Mass Bounds\nProves: Not Ruled Out', fillcolor='lightcoral')

# Edges for vertical flow
dot.edge('Central', 'Anomalies', label='Motivation')
dot.edge('Anomalies', 'Algebra', label='Algebraic Foundation')
dot.edge('Algebra', 'Coupling', label='Derive Interaction')
dot.edge('Coupling', 'Renorm', label='In-Medium Effects')
dot.edge('Renorm', 'Naturalness', label='Resolve Hierarchy')
dot.edge('Naturalness', 'Condensate', label='Critical Condensate')
dot.edge('Condensate', 'Validation', label='Quantitative Predictions')
dot.edge('Validation', 'Discrim', label='Testable Differences')
dot.edge('Discrim', 'Constraints', label='Bound Consistency')
dot.edge('Constraints', 'Central', label='Closed Loop', style='dashed', color='gray')

# Render and save as PDF
dot.render('z3_chapter1_mindmap_vertical', view=True, cleanup=True)
print("Vertical mindmap generated and saved as z3_chapter1_mindmap_vertical.pdf")
print("Layout: Top to bottom; key nodes colored for reviewer concerns (formulas/proofs highlighted).")
print("Perfect for LaTeX inclusion (tall format fits page).")