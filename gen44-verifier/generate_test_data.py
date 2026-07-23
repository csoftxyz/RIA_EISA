"""
gen44 Terminal Certificate Verifier v0.9 — 测试数据生成器

生成多个测试场景:
1. perfect_44 — 结构完美的 44 终端
2. undercount_43 — 只有 43 个等价类 (应 FAIL)
3. duplicate_classes — 等价类重复 (应 FAIL)
4. unreachable_class — 存在不可达终端类 (应 FAIL)
5. closure_failure — 闭包不完整 (应 FAIL)
6. triality_violation — T³ ≠ id (应 FAIL)
7. congruence_violation — 等价前提产生不同结论 (应 FAIL)
8. schur_violation — 跨块非容许 Hom (应 FAIL)
9. confluence_certified — 带 certified critical pairs
"""

from models import (
    ProofBundle, EquivalenceClass, RuleInstance,
    DerivationStep, MergeEvent, CriticalPair, RuleType
)
from typing import List, Dict, Optional
import json
import os


# ─────────────────────────────────────
# 基础构建工具
# ─────────────────────────────────────

def _make_perfect_44_base() -> ProofBundle:
    """构建一个完美的 44 终端证明包 — 所有检查应 PASS"""
    bundle = ProofBundle(
        bundle_id="gen44-v0.9-perfect-44",
        description="Structurally perfect 44 terminal — baseline test",
    )

    # Seed: 2 classes
    SD = 1  # seed dark
    SL = 2  # seed light
    seeds = [SD, SL]

    # 终端等价类: 44 classes with structured attributes
    classes = {}
    terminal = []

    # 14 triality orbits × 3 + 2 fixed points = 42 + 2 = 44
    # Orbit 0: 3 classes, shell 0, block B0
    # Orbit 1: 3 classes, shell 0, block B1
    # ... up to 44

    cid = 1
    orbit_id = 0
    shell_id = 0
    block_letter = 'A'
    orbit_indices = []

    for i in range(44):
        if i < 42:
            # Triality orbit of size 3
            if i % 3 == 0:
                orbit_id = i // 3
                shell_id = orbit_id % 7  # 7 shells
                block_letter = chr(ord('A') + (orbit_id % 10))
                orbit_indices = []

            grade = i % 3
            orbit_indices.append(cid)

            block = f"B{block_letter}" if i < 40 else f"B{chr(ord('K') + (i - 40))}"
            schur = "admissible" if i % 5 != 0 else "scalar"
        else:
            # Fixed points (T³ = id, T(x) = x), shells 6
            grade = 0
            block = f"B{chr(ord('A') + (i - 42))}"
            schur = "scalar"
            orbit_indices = [cid]

        eq_class = EquivalenceClass(
            class_id=cid,
            canonical_rep=f"t_{cid:02d}",
            members=[f"t_{cid:02d}"],
            shell=shell_id if i < 42 else 6,
            triality_orbit=orbit_id if (i < 42 and i % 3 == 0) else (
                orbit_id if i < 42 else 14 + (i - 42)
            ),
            triality_grade=grade,
            block=block,
            schur_type=schur,
        )
        classes[cid] = eq_class
        terminal.append(cid)
        bundle.parent[cid] = cid

        cid += 1

        # Record triality orbit at boundary
        if i < 42 and len(orbit_indices) == 3:
            bundle.triality_orbits.append(list(orbit_indices))
        elif i >= 42:
            bundle.triality_orbits.append(list(orbit_indices))

    bundle.classes = classes
    bundle.terminal_classes = terminal
    bundle.seed_classes = seeds

    # Build rule instances: triality, shell, and tensor rules
    ri_count = 0

    # Triality rules: each orbit group
    for orbit in bundle.triality_orbits:
        if len(orbit) == 3:
            # T: orbit[0] → orbit[1], T: orbit[1] → orbit[2], T: orbit[2] → orbit[0]
            for src_idx in range(3):
                tgt = orbit[(src_idx + 1) % 3]
                ri = RuleInstance(
                    instance_id=f"INST-{ri_count:04d}",
                    rule_type="triality",
                    premises=[orbit[src_idx]],
                    conclusions=[tgt],
                )
                bundle.rule_instances.append(ri)
                ri_count += 1
        else:
            # Fixed point: T(x) = x
            ri = RuleInstance(
                instance_id=f"INST-{ri_count:04d}",
                rule_type="triality",
                premises=[orbit[0]],
                conclusions=[orbit[0]],
            )
            bundle.rule_instances.append(ri)
            ri_count += 1

    # Shell rules: only within same shell (conserved)
    for i in range(0, 42, 3):
        for grade_a in range(3):
            for grade_b in range(3):
                if grade_a != grade_b:
                    src = terminal[i + grade_a]
                    tgt = terminal[i + grade_b]
                    ri = RuleInstance(
                        instance_id=f"INST-{ri_count:04d}",
                        rule_type="shell",
                        premises=[src],
                        conclusions=[tgt],
                    )
                    bundle.rule_instances.append(ri)
                    ri_count += 1

    # Wedderburn rules: block-closed
    # Group classes by block
    block_groups: Dict[str, List[int]] = {}
    for cid in terminal:
        blk = classes[cid].block
        if blk not in block_groups:
            block_groups[blk] = []
        block_groups[blk].append(cid)

    for blk, members in block_groups.items():
        if len(members) >= 2:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    # Closure: (x, y) → z (some member in same block)
                    z = members[(i + j) % len(members)]
                    ri = RuleInstance(
                        instance_id=f"INST-{ri_count:04d}",
                        rule_type="wedderburn",
                        premises=[members[i], members[j]],
                        conclusions=[z],
                    )
                    bundle.rule_instances.append(ri)
                    ri_count += 1

    # Derivation steps: from seed, derive each terminal class via proper DAG
    # Strategy: chain within each triality orbit, starting from previous orbit
    for seed in seeds:
        ds = DerivationStep(
            target_class=seed,
            rule_instance_id="none",
            parent_classes=[],
            depth=0,
        )
        bundle.derivation_steps.append(ds)

    # Build orbit chains: each orbit[1] from orbit[0], orbit[2] from orbit[1].
    # For orbit[0] of non-seed orbits: bootstrap from a reachable class.
    prev_reachable = list(seeds)

    for orbit_idx, orbit in enumerate(bundle.triality_orbits):
        # Check if this orbit already has seed classes
        has_seed = bool(set(orbit) & set(seeds))

        for idx in range(len(orbit)):
            target = orbit[idx]
            if target in seeds:
                prev_reachable.append(target)
                continue

            if idx == 0 and not has_seed:
                # Bootstrap: first element of non-seed orbit needs external entry
                # Use any reachable class as bootstrap parent
                ds = DerivationStep(
                    target_class=target,
                    rule_instance_id="orbit-bootstrap",
                    parent_classes=[seeds[0]],
                    depth=1,
                )
                bundle.derivation_steps.append(ds)
                prev_reachable.append(target)

            elif idx > 0:
                # Chain from previous element in same orbit
                parent = orbit[idx - 1]
                for inst in bundle.rule_instances:
                    if (target in inst.conclusions and
                        inst.rule_type == RuleType.TRIALITY.value and
                        parent in inst.premises):
                        ds = DerivationStep(
                            target_class=target,
                            rule_instance_id=inst.instance_id,
                            parent_classes=[parent],
                            depth=1,
                        )
                        bundle.derivation_steps.append(ds)
                        prev_reachable.append(target)
                        break
                else:
                    # Fallback: direct chain
                    ds = DerivationStep(
                        target_class=target,
                        rule_instance_id="orbit-chain",
                        parent_classes=[parent],
                        depth=1,
                    )
                    bundle.derivation_steps.append(ds)
                    prev_reachable.append(target)
            else:
                # idx == 0 and has_seed: this element is a seed, already handled
                pass

    # Normal forms (identity)
    for cid in terminal:
        bundle.normal_forms[f"t_{cid:02d}"] = f"t_{cid:02d}"

    # Invariants
    shell_counts = {}
    for cid in terminal:
        sh = classes[cid].shell
        shell_counts[sh] = shell_counts.get(sh, 0) + 1
    bundle.shell_counts = shell_counts

    tri_orbit_sizes = {3: 14, 1: 2}  # 14×3 + 2×1 = 44
    bundle.triality_orbit_sizes = tri_orbit_sizes

    block_counts = {}
    for cid in terminal:
        blk = classes[cid].block
        block_counts[blk] = block_counts.get(blk, 0) + 1
    bundle.block_counts = block_counts

    bundle.invariants = {
        "total_classes": 44,
        "num_seeds": 2,
        "num_rule_instances": ri_count,
        "num_filtered": 0,
        "num_merges": 0,
    }

    bundle.termination = {
        "universe_size": 44,
        "max_shell": 6,
        "num_blocks": len(block_counts),
    }

    return bundle


# ─────────────────────────────────────
# 测试场景生成器
# ─────────────────────────────────────

def perfect_44() -> ProofBundle:
    """场景 1: 结构完美的 44 终端 — 应全部 PASS"""
    bundle = _make_perfect_44_base()
    bundle.description = "Structurally perfect 44 — all checks should PASS"

    # Add critical pairs for confluence
    for i in range(3):
        cp = CriticalPair(
            pair_id=f"CP-{i:03d}",
            source_term=f"t_{i+1:02d}",
            rule_a="triality",
            result_a=(i + 2) if i < 2 else 1,
            rule_b="shell",
            result_b=i + 2 if i < 2 else 1,
            join_witness=i + 2 if i < 2 else 1,
        )
        bundle.critical_pairs.append(cp)

    return bundle


def undercount_43() -> ProofBundle:
    """场景 2: 只有 43 个等价类 — count_44 应 FAIL"""
    bundle = _make_perfect_44_base()
    bundle.bundle_id = "gen44-v0.9-undercount-43"
    bundle.description = "43 terminal classes — count_44 should FAIL"

    # Remove last class (save reference before deleting)
    last = bundle.terminal_classes[-1]
    last_cls = bundle.classes.get(last)
    last_shell = last_cls.shell if last_cls else 0

    bundle.terminal_classes = bundle.terminal_classes[:-1]
    if last in bundle.classes:
        del bundle.classes[last]

    # Also remove last triality orbit
    if bundle.triality_orbits:
        bundle.triality_orbits = bundle.triality_orbits[:-1]

    # Adjust shell counts
    new_counts = {}
    for sh, cnt in bundle.shell_counts.items():
        new_counts[sh] = cnt - 1 if sh == last_shell else cnt
    bundle.shell_counts = {k: v for k, v in new_counts.items() if v > 0}

    return bundle


def duplicate_classes() -> ProofBundle:
    """场景 3: 等价类重复 — no_duplicates 应 FAIL"""
    bundle = _make_perfect_44_base()
    bundle.bundle_id = "gen44-v0.9-duplicate"
    bundle.description = "Duplicate equivalence class — no_duplicates should FAIL"

    # Duplicate class 1 into terminal (same root)
    dup = bundle.terminal_classes[1]
    bundle.terminal_classes.insert(0, dup)
    bundle.parent[dup] = bundle.parent[bundle.terminal_classes[0]]

    return bundle


def unreachable_class() -> ProofBundle:
    """场景 4: 存在不可达终端类 — reachability 应 FAIL"""
    bundle = _make_perfect_44_base()
    bundle.bundle_id = "gen44-v0.9-unreachable"
    bundle.description = "Unreachable terminal class — reachability should FAIL"

    # Add an extra terminal class with no derivation
    new_id = 100
    bundle.classes[new_id] = EquivalenceClass(
        class_id=new_id,
        canonical_rep="t_orphan",
        members=["t_orphan"],
        shell=9,
        triality_grade=0,
        block="B_orphan",
        schur_type="admissible",
    )
    bundle.parent[new_id] = new_id
    bundle.terminal_classes.append(new_id)

    return bundle


def closure_failure() -> ProofBundle:
    """场景 5: 闭包不完整 — closure_mod_equiv 应 FAIL"""
    bundle = _make_perfect_44_base()
    bundle.bundle_id = "gen44-v0.9-closure-fail"
    bundle.description = "Incomplete closure — closure_mod_equiv should FAIL"

    # Add a rule instance whose premises are all in terminal
    # but whose conclusion is NOT in terminal
    new_conc = 200
    premises = [bundle.terminal_classes[0], bundle.terminal_classes[1]]
    ri = RuleInstance(
        instance_id="INST-CLO-FAIL",
        rule_type="wedderburn",
        premises=premises,
        conclusions=[new_conc],
    )
    bundle.rule_instances.append(ri)
    bundle.parent[new_conc] = new_conc  # not in terminal_classes

    return bundle


def triality_violation() -> ProofBundle:
    """场景 6: T³ ≠ id — triality 应 FAIL"""
    bundle = _make_perfect_44_base()
    bundle.bundle_id = "gen44-v0.9-triality-fail"
    bundle.description = "T³ ≠ id violation — triality should FAIL"

    # Add an invalid orbit (size 2)
    bundle.triality_orbits.append([1, 2])  # bad size
    bundle.triality_orbit_sizes = {3: 14, 1: 2, 2: 1}

    return bundle


def schur_violation() -> ProofBundle:
    """场景 7: 跨块非容许 Hom — schur 应 FAIL"""
    bundle = _make_perfect_44_base()
    bundle.bundle_id = "gen44-v0.9-schur-fail"
    bundle.description = "Schur violation — schur should FAIL"

    # Add wedderburn rule that crosses blocks
    blks = list(bundle.block_counts.keys())
    if len(blks) >= 2:
        # Find classes from different blocks
        a_class = None
        b_class = None
        for cid in bundle.terminal_classes:
            cls = bundle.classes[cid]
            if cls.block == blks[0] and a_class is None:
                a_class = cid
            if cls.block == blks[1] and b_class is None:
                b_class = cid

        if a_class and b_class:
            target = bundle.terminal_classes[len(bundle.terminal_classes) // 2]
            ri = RuleInstance(
                instance_id="INST-SCHUR-FAIL",
                rule_type="wedderburn",
                premises=[a_class, b_class],
                conclusions=[target],
            )
            bundle.rule_instances.append(ri)

    return bundle


def congruence_violation() -> ProofBundle:
    """场景 8: 等价前提产生不同结论 — rule_congruence 应 FAIL"""
    bundle = _make_perfect_44_base()
    bundle.bundle_id = "gen44-v0.9-congruence-fail"
    bundle.description = "Congruence violation — rule_congruence should FAIL"

    # Merge two classes
    c1 = bundle.terminal_classes[0]
    c2 = bundle.terminal_classes[1]
    bundle.union(c1, c2)

    # Now add two rule instances with same (now equivalent) premises
    # but different conclusion roots that are NOT merged
    root = bundle.find(c1)
    c3 = bundle.terminal_classes[3]
    c4 = bundle.terminal_classes[4]

    ri1 = RuleInstance(
        instance_id="INST-CONG-A",
        rule_type="shell",
        premises=[root],
        conclusions=[c3],
    )
    ri2 = RuleInstance(
        instance_id="INST-CONG-B",
        rule_type="shell",
        premises=[root],
        conclusions=[c4],
    )
    bundle.rule_instances.append(ri1)
    bundle.rule_instances.append(ri2)

    return bundle


def confluence_certified() -> ProofBundle:
    """场景 9: 带完整 critical pair 证书 — confluence 应 PASS"""
    bundle = _make_perfect_44_base()
    bundle.bundle_id = "gen44-v0.9-confluence"
    bundle.description = "Full confluence certificate — confluence should PASS"

    # All possible critical pair pairs between triality and shell
    # on the first orbit
    orbit0 = bundle.triality_orbits[0] if bundle.triality_orbits else []
    if len(orbit0) >= 3:
        for i, src in enumerate(orbit0):
            # T(source) vs shell(source)
            tgt_t = orbit0[(i + 1) % 3]
            # Shell to next grade in same orbit
            tgt_s = orbit0[(i + 2) % 3] if i % 3 != 2 else orbit0[0]
            # They join at orbit0[(i+1)%3] (both T and shell can reach it)
            join = orbit0[(i + 1) % 3]

            cp = CriticalPair(
                pair_id=f"CP-{i:03d}",
                source_term=f"t_{src:02d}",
                rule_a="triality",
                result_a=tgt_t,
                rule_b="shell",
                result_b=tgt_s,
                join_witness=join,
            )
            bundle.critical_pairs.append(cp)

    return bundle


def gen44_sample_with_log() -> ProofBundle:
    """
    从模拟的 gen44 程序日志构建证明包 — 端到端 PASS 测试 (P0 fixed).

    44 = 14 triality orbits × 3 + 2 fixed points.
    Seed: t_000, t_001 (explicitly declared).
    Bootstrap rules connect orbit entry points.
    """
    from compiler import ProofCompiler

    log = []
    log.append("# gen44 program log — end-to-end PASS case (P0 fixed)")
    log.append("")

    # 42 triality terms (14 orbits × 3)
    for i in range(42):
        orbit = i // 3
        grade = i % 3
        shell = orbit % 7
        block = f"B{chr(ord('A') + (orbit % 10))}"
        schur = "admissible" if grade != 0 else "scalar"
        log.append(
            f"add t_{i:03d} shell={shell} grade={grade} "
            f"block={block} schur={schur}"
        )

    # 2 fixed points (T(x) = x)
    log.append("add t_fix0 shell=6 grade=0 block=B_FIX0 schur=scalar")
    log.append("add t_fix1 shell=6 grade=0 block=B_FIX1 schur=scalar")
    log.append("")

    # Explicit seeds
    log.append("seed t_000")
    log.append("seed t_001")
    log.append("")

    # Triality rules (within each orbit)
    for i in range(42):
        base = i - (i % 3)
        nxt = base + ((i + 1) % 3)
        log.append(f"fire triality t_{i:03d} -> t_{nxt:03d}")

    log.append("fire triality t_fix0 -> t_fix0")
    log.append("fire triality t_fix1 -> t_fix1")
    log.append("")

    # Bootstrap reachability: connect seed to each orbit's first element
    # and to fixed points
    for orbit in range(1, 14):
        target = orbit * 3
        log.append(f"fire bootstrap t_000 -> t_{target:03d}")

    log.append("fire bootstrap t_000 -> t_fix0")
    log.append("fire bootstrap t_000 -> t_fix1")
    log.append("")

    # Shell rules inside each orbit
    for orbit in range(14):
        for a in range(3):
            for b in range(3):
                if a != b:
                    src = orbit * 3 + a
                    tgt = orbit * 3 + b
                    log.append(f"fire shell t_{src:03d} -> t_{tgt:03d}")

    log.append("")

    # Wedderburn rules inside each orbit/block
    for orbit in range(14):
        o = orbit * 3
        log.append(f"fire wedderburn t_{o:03d},t_{o+1:03d} -> t_{o+2:03d}")
        log.append(f"fire wedderburn t_{o+1:03d},t_{o+2:03d} -> t_{o:03d}")

    log.append("")
    log.append("terminal size 44")

    compiler = ProofCompiler()
    return compiler.compile_from_log(log)


# ─────────────────────────────────────
# 批量生成并保存
# ─────────────────────────────────────

ALL_SCENARIOS = {
    "perfect_44": perfect_44,
    "undercount_43": undercount_43,
    "duplicate_classes": duplicate_classes,
    "unreachable_class": unreachable_class,
    "closure_failure": closure_failure,
    "triality_violation": triality_violation,
    "schur_violation": schur_violation,
    "congruence_violation": congruence_violation,
    "confluence_certified": confluence_certified,
    "gen44_sample": gen44_sample_with_log,
}


def generate_all(output_dir: str = "/tmp/gen44-test-bundles"):
    """生成所有测试数据并保存"""
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for name, factory in ALL_SCENARIOS.items():
        path = os.path.join(output_dir, f"{name}.json")
        bundle = factory()
        bundle.save(path)
        results[name] = {
            "path": path,
            "description": bundle.description,
            "terminal_size": len(bundle.terminal_classes),
            "num_rules": len(bundle.rule_instances),
            "num_critical_pairs": len(bundle.critical_pairs),
        }
        print(f"  Generated: {name} ({len(bundle.terminal_classes)} classes, "
              f"{len(bundle.rule_instances)} rules, "
              f"{len(bundle.critical_pairs)} CPs) -> {path}")

    return results


if __name__ == "__main__":
    print("=" * 64)
    print("  gen44 Test Data Generator")
    print("=" * 64)
    results = generate_all()
    print(f"\nGenerated {len(results)} test bundles.")
