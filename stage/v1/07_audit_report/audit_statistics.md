# Audit & Project Statistics

**Project start:** February 2025 · **Report date:** November 2026 · **Package:** v24

## Audit activity
| Item | Count |
|---|---|
| External audits (论文审计1–13) | 13 |
| Self-audits (physicist, programmer, v23, v24×2, + round-3 code) | 6 rounds (5 report files at project root; the v23 report covers two rounds) |
| Mining directions pursued | 8 |
| Headline conclusions withdrawn & rewritten | 3 (0→4→3→0) |
| Real code bugs found & fixed | 5 (derivative seed, orbit subtraction, hard-coded assertions, free/tied v_L, gen rank-12) |
| Misattributions caught by controls | 2 (V1, V7) |

## Deliverable documents (5)
| Document | Pages |
|---|---|
| Cover letter | 4 |
| Main paper | 14 |
| Supplemental Material | 12 |
| Code documentation | 61 |
| Audit report (this folder, PDF) | see AUDIT_REPORT.pdf |

## Code
| Location | Files | Lines |
|---|---|---|
| 04_verification_code (active, top-level) | 33 | 8,035 |
| 04_verification_code/tests_validation | 23 | 3,630 |
| 04_verification_code/legacy (withdrawn, kept for provenance) | 7 | 947 |
| 08_gui (control panel + screenshot capture) | 2 | 383 |
| **Total** | **65** | **12,995** |

## Verification
- Master verifier assertions: **70 / 70 pass**.
- Figures: 12 (9 wide + single-column renditions) + 8 GUI button icons + 1 journey flowchart + 2 GUI screenshots.
- Locked environment: Python 3.6.6 / numpy 1.19.5 / scipy 1.5.4 / matplotlib 3.3.4.
- Truth source deterministic; basis-independence verified to ~1e-14.

## Working-report archive (project root, Chinese)
50 Markdown working reports (including 5 self-audit reports) + 13 external-audit text files
document every round in detail; this English audit report consolidates them.
