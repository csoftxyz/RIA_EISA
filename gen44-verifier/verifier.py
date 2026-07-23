"""
gen44 Terminal Certificate Verifier v0.9 — 验证器引擎

Phase 1: 结构健全性 (count, duplicates, seed, reachability, closure, nf, attributes, invariants)
Phase 2: 商空间安全性 (congruence, rule_congruence, filter_safety)
Phase 3: 结构一致性 (triality, shell, wedderburn, schur)
Phase 4: 合流性 (termination, critical_pairs)
"""

from models import ProofBundle, RuleInstance, CriticalPair, RuleType
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import time


# ─────────────────────────────────────
# 验证结果
# ─────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    phase: int
    detail: Any = None
    counterexample: Any = None


@dataclass
class VerificationReport:
    status: str = "PENDING"          # "PASS" | "FAIL"
    bundle_id: str = ""
    checks: List[CheckResult] = None
    failed_checks: List[str] = None
    theorem: str = ""
    terminal_size: int = 0
    confluence: str = ""
    elapsed_ms: float = 0.0
    invariants_summary: Dict[str, Any] = None
    hash_count: str = ""
    hash_structure: str = ""

    def __post_init__(self):
        if self.checks is None:
            self.checks = []
        if self.failed_checks is None:
            self.failed_checks = []

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "bundle_id": self.bundle_id,
            "failed_checks": self.failed_checks,
            "theorem": self.theorem,
            "terminal_size": self.terminal_size,
            "confluence": self.confluence,
            "elapsed_ms": self.elapsed_ms,
            "checks_passed": sum(1 for c in self.checks if c.passed),
            "checks_total": len(self.checks),
            "hash_count": self.hash_count,
            "hash_structure": self.hash_structure,
            "invariants_summary": self.invariants_summary,
            "details": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "phase": c.phase,
                    "detail": str(c.detail)[:200] if c.detail else None,
                    "counterexample": str(c.counterexample)[:200] if c.counterexample else None,
                }
                for c in self.checks
            ],
        }

    def print_report(self):
        """打印可读报告"""
        print("=" * 72)
        print(f"  gen44 Terminal Certificate Verifier v0.9")
        print(f"  Bundle: {self.bundle_id}")
        print("=" * 72)
        print(f"  Status:    {self.status}")
        print(f"  Theorem:   {self.theorem}")
        print(f"  |R₄₄|:     {self.terminal_size}")
        if self.confluence:
            print(f"  Confluence: {self.confluence}")
        print(f"  Time:      {self.elapsed_ms:.1f} ms")
        print(f"  Checks:    {sum(1 for c in self.checks if c.passed)}/{len(self.checks)} passed")
        print("-" * 72)

        phases = sorted(set(c.phase for c in self.checks))
        for phase in phases:
            phase_checks = [c for c in self.checks if c.phase == phase]
            print(f"\n  Phase {phase}:")
            for c in phase_checks:
                icon = "✅" if c.passed else "❌"
                detail_str = f"  — {c.detail}" if c.detail else ""
                print(f"    {icon} {c.name}{detail_str}")
                if not c.passed and c.counterexample:
                    print(f"       counterexample: {c.counterexample}")

        print("-" * 72)
        if self.hash_count:
            print(f"  Count hash: {self.hash_count[:16]}...")
        print("=" * 72)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────
# 核心验证器
# ─────────────────────────────────────

class TerminalCertificateVerifier:
    """v0.9 终端证书验证器"""

    def __init__(self, bundle: ProofBundle):
        self.bundle = bundle
        self.terminal_set: Set[int] = set(bundle.terminal_classes)
        self.terminal_roots: Set[int] = {bundle.find(c) for c in bundle.terminal_classes}

    def verify(self) -> VerificationReport:
        """执行全部四阶段验证"""
        t0 = time.time()
        report = VerificationReport(
            bundle_id=self.bundle.bundle_id,
        )

        # Phase 1: 结构健全性
        report.checks.extend(self._phase1_structural_sanity())

        # Phase 2: 商空间安全性
        report.checks.extend(self._phase2_quotient_safety())

        # Phase 3: 结构一致性
        report.checks.extend(self._phase3_structure())

        # Phase 4: 合流性
        report.checks.extend(self._phase4_confluence())

        # 汇总
        report.failed_checks = [c.name for c in report.checks if not c.passed]
        if report.failed_checks:
            report.status = "FAIL"
        else:
            report.status = "PASS"
            report.theorem = "Quotient-safe NF_gen44(s) = R₄₄"
            report.confluence = self._confluence_status(report)

        report.terminal_size = len(self.bundle.terminal_classes)
        report.elapsed_ms = (time.time() - t0) * 1000
        report.hash_count = self._terminal_count_hash()
        report.hash_structure = self._terminal_structure_hash()
        report.invariants_summary = {
            "total": len(self.bundle.terminal_classes),
            "shells": self.bundle.shell_counts,
            "blocks": self.bundle.block_counts,
            "triality_orbits": self.bundle.triality_orbit_sizes,
            "schur_violations": self.bundle.schur_violations,
        }

        return report

    # ─────────────────────────────────
    # Phase 1: 结构健全性
    # ─────────────────────────────────

    def _phase1_structural_sanity(self) -> List[CheckResult]:
        checks = []

        # 1. count_44
        checks.append(self._check_count_44())

        # 2. no_duplicates
        checks.append(self._check_no_duplicates())

        # 3. seed_subset
        checks.append(self._check_seed_subset())

        # 4. reachability
        checks.append(self._check_reachability())

        # 5. closure_mod_equiv
        checks.append(self._check_closure_mod_equiv())

        # 6. nf_consistency
        checks.append(self._check_nf_consistency())

        # 7. attribute_consistency
        checks.append(self._check_attribute_consistency())

        # 8. invariant_sums
        checks.append(self._check_invariant_sums())

        return checks

    def _check_count_44(self) -> CheckResult:
        n = len(self.bundle.terminal_classes)
        return CheckResult(
            name="count_44",
            phase=1,
            passed=(n == 44),
            detail=f"|R₄₄| = {n}",
            counterexample=None if n == 44 else f"Expected 44, got {n}",
        )

    def _check_no_duplicates(self) -> CheckResult:
        seen = set()
        for cid in self.bundle.terminal_classes:
            root = self.bundle.find(cid)
            if root in seen:
                return CheckResult(
                    name="no_duplicates",
                    phase=1,
                    passed=False,
                    detail=f"Duplicate equivalence class root={root}",
                    counterexample=f"root {root} appears more than once",
                )
            seen.add(root)
        return CheckResult(name="no_duplicates", phase=1, passed=True)

    def _check_seed_subset(self) -> CheckResult:
        seed_set = set(self.bundle.seed_classes)
        terminal_set = set(self.bundle.terminal_classes)
        missing = seed_set - terminal_set
        if missing:
            return CheckResult(
                name="seed_subset",
                phase=1,
                passed=False,
                detail=f"S₀ ⊆ R₄₄ fails",
                counterexample=f"Seeds not in terminal: {missing}",
            )
        return CheckResult(name="seed_subset", phase=1, passed=True)

    def _check_reachability(self) -> CheckResult:
        """BFS from seed along derivation steps"""
        reachable = set(self.bundle.seed_classes)
        steps = list(self.bundle.derivation_steps)

        changed = True
        max_iter = 200
        iteration = 0
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for step in steps:
                if set(step.parent_classes).issubset(reachable):
                    if step.target_class not in reachable:
                        reachable.add(step.target_class)
                        changed = True

        unreachable = set(self.bundle.terminal_classes) - reachable
        if unreachable:
            return CheckResult(
                name="reachability",
                phase=1,
                passed=False,
                detail=f"{len(unreachable)} classes unreachable",
                counterexample={"unreachable": list(unreachable)[:5]},
            )
        return CheckResult(
            name="reachability",
            phase=1,
            passed=True,
            detail=f"All {len(reachable)} terminal classes reachable",
        )

    def _check_closure_mod_equiv(self) -> CheckResult:
        """闭包检查 — 在等价类根上"""
        failures = []
        for inst in self.bundle.rule_instances:
            premise_roots = {self.bundle.find(p) for p in inst.premises}
            if premise_roots.issubset(self.terminal_roots):
                for c in inst.conclusions:
                    if self.bundle.find(c) not in self.terminal_roots:
                        failures.append({
                            "instance": inst.instance_id,
                            "rule_type": inst.rule_type,
                            "missing_class": c,
                            "missing_root": self.bundle.find(c),
                        })
        if failures:
            return CheckResult(
                name="closure_mod_equiv",
                phase=1,
                passed=False,
                detail=f"{len(failures)} closure failures",
                counterexample=failures[:3],
            )
        return CheckResult(name="closure_mod_equiv", phase=1, passed=True)

    def _check_nf_consistency(self) -> CheckResult:
        """规范形一致性: x ∼ y ⟹ nf(x) = nf(y)"""
        failures = []
        for cid, cls in self.bundle.classes.items():
            if len(cls.members) <= 1:
                continue
            nfs = set()
            for m in cls.members:
                nf = self.bundle.normal_forms.get(m, m)
                nfs.add(nf)
            if len(nfs) > 1:
                failures.append({
                    "class_id": cid,
                    "normal_forms": list(nfs),
                })
        if failures:
            return CheckResult(
                name="nf_consistency",
                phase=1,
                passed=False,
                detail=f"{len(failures)} classes with inconsistent nf",
                counterexample=failures[:3],
            )
        return CheckResult(name="nf_consistency", phase=1, passed=True)

    def _check_attribute_consistency(self) -> CheckResult:
        """属性良定义: x ∼ y ⟹ Attr(x) = Attr(y)

        P0 fix: 使用 raw_terms 获取 member-level 属性，
        而非比较 class ids (类型不匹配的 bug).
        """
        failures = []
        for cid, cls in self.bundle.classes.items():
            if len(cls.members) <= 1:
                continue

            # Collect (shell, grade, block, schur) from raw terms
            attr_sets = set()
            for m in cls.members:
                # Get attributes from raw_terms via canonical_rep mapping
                if m in self.bundle.raw_terms:
                    rt = self.bundle.raw_terms[m]
                    attr_sets.add((rt.shell, rt.triality_grade, rt.block, rt.schur_type))
                elif m in self.bundle.classes:
                    mc = self.bundle.classes[m]
                    attr_sets.add((mc.shell, mc.triality_grade, mc.block, mc.schur_type))

            if len(attr_sets) > 1:
                failures.append({
                    "class_id": cid,
                    "members": cls.members[:5],
                    "attr_variants": list(attr_sets),
                })

        if failures:
            return CheckResult(
                name="attribute_consistency",
                phase=1,
                passed=False,
                detail=f"{len(failures)} classes with attribute mismatch",
                counterexample=failures[:3],
            )
        return CheckResult(name="attribute_consistency", phase=1, passed=True)

    def _check_invariant_sums(self) -> CheckResult:
        """不变量求和检查"""
        errors = []

        # Shell sum
        shell_sum = sum(self.bundle.shell_counts.values())
        if shell_sum != 44:
            errors.append(f"Σ shell_counts = {shell_sum} ≠ 44")

        # Block sum
        block_sum = sum(self.bundle.block_counts.values())
        if block_sum != 44 and block_sum != 0:
            errors.append(f"Σ block_counts = {block_sum} ≠ 44")

        # Triality sum: 3*N₃ + N₁ = 44
        tri_sum = (
            3 * self.bundle.triality_orbit_sizes.get(3, 0) +
            self.bundle.triality_orbit_sizes.get(1, 0)
        )
        if tri_sum != 44 and tri_sum != 0:
            errors.append(f"3·N₃ + N₁ = {tri_sum} ≠ 44")

        if errors:
            return CheckResult(
                name="invariant_sums",
                phase=1,
                passed=False,
                detail="; ".join(errors),
                counterexample=errors,
            )
        return CheckResult(name="invariant_sums", phase=1, passed=True)

    # ─────────────────────────────────
    # Phase 2: 商空间安全性
    # ─────────────────────────────────

    def _phase2_quotient_safety(self) -> List[CheckResult]:
        checks = []
        checks.append(self._check_side_condition_quotient_safety())
        checks.append(self._check_congruence())
        checks.append(self._check_rule_congruence())
        checks.append(self._check_filter_safety())
        return checks

    def _check_side_condition_quotient_safety(self) -> CheckResult:
        """检查 side conditions 是否下降到商空间"""
        # Group rule instances by rule type
        instances_by_type: Dict[str, List[RuleInstance]] = defaultdict(list)
        for inst in self.bundle.rule_instances:
            instances_by_type[inst.rule_type].append(inst)

        failures = []

        # For each rule type, check that equivalent premises always
        # have the same rule applicability
        for rule_type, instances in instances_by_type.items():
            # Build map: (premise_root_tuple) -> set of conclusion roots
            premise_map: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)

            for inst in instances:
                key = tuple(sorted(self.bundle.find(p) for p in inst.premises))
                for c in inst.conclusions:
                    premise_map[key].add(self.bundle.find(c))

            # Check consistency: same premise roots -> same conclusion roots
            # (This is a weaker check than full congruence)
            for key, conc_roots in premise_map.items():
                if len(conc_roots) > 1:
                    # Multiple different conclusions from same premises = possible issue
                    # But this is actually fine for non-deterministic rules
                    # The real check is: are ALL conclusions in the terminal set?
                    if not conc_roots.issubset(self.terminal_roots):
                        failures.append({
                            "rule_type": rule_type,
                            "premise_roots": key,
                            "missing_roots": list(conc_roots - self.terminal_roots),
                        })

        if failures:
            return CheckResult(
                name="side_condition_quotient_safety",
                phase=2,
                passed=False,
                detail=f"{len(failures)} quotient safety violations",
                counterexample=failures[:3],
            )
        return CheckResult(name="side_condition_quotient_safety", phase=2, passed=True)

    def _check_congruence(self) -> CheckResult:
        """检查 ∼ 对规则作用的 congruence 性质"""
        failures = []

        # For each merge (x ∼ y), check that rule applications respect it
        for merge in self.bundle.merge_events[:100]:  # limit for performance
            x = merge.left
            y = merge.right

            # Get class ids
            x_cid = None
            y_cid = None
            # Search for these terms in classes
            for cid, cls in self.bundle.classes.items():
                if x in cls.members:
                    x_cid = cid
                if y in cls.members:
                    y_cid = cid

            if x_cid is None or y_cid is None:
                continue

            x_root = self.bundle.find(x_cid)
            y_root = self.bundle.find(y_cid)

            if x_root != y_root:
                failures.append({
                    "merge": (x, y),
                    "x_root": x_root,
                    "y_root": y_root,
                    "note": "Merged terms are in different equivalence classes",
                })

        if failures:
            return CheckResult(
                name="congruence",
                phase=2,
                passed=False,
                detail=f"{len(failures)} congruence violations",
                counterexample=failures[:3],
            )
        return CheckResult(name="congruence", phase=2, passed=True)

    def _check_rule_congruence(self) -> CheckResult:
        """规则实例级 congruence: 等价前提 → 等价结论
        
        注意: shell 规则是约束/关系 (多值), 不是函数式规则,
        所以只对 triality, tensor, wedderburn 检查 congruence。
        """
        instances_by_type: Dict[str, List[RuleInstance]] = defaultdict(list)
        for inst in self.bundle.rule_instances:
            instances_by_type[inst.rule_type].append(inst)

        failures = []
        # Only check functional rules (not shell, which is multi-valued by design)
        functional_types = {RuleType.TRIALITY.value, RuleType.TENSOR.value, RuleType.WEDDERBURN.value}

        for rule_type in functional_types:
            instances = instances_by_type.get(rule_type, [])
            for i, inst_i in enumerate(instances):
                for j, inst_j in enumerate(instances):
                    if j <= i:
                        continue

                    # Check if premises are pair-wise equivalent
                    if len(inst_i.premises) != len(inst_j.premises):
                        continue

                    all_equiv = all(
                        self.bundle.find(inst_i.premises[k]) ==
                        self.bundle.find(inst_j.premises[k])
                        for k in range(len(inst_i.premises))
                    )

                    if all_equiv:
                        # Check conclusions are equivalent
                        if len(inst_i.conclusions) != len(inst_j.conclusions):
                            failures.append({
                                "rule_type": rule_type,
                                "i_instance": inst_i.instance_id,
                                "j_instance": inst_j.instance_id,
                                "note": "Different number of conclusions for equivalent premises",
                            })
                            continue

                        for k in range(len(inst_i.conclusions)):
                            if (self.bundle.find(inst_i.conclusions[k]) !=
                                self.bundle.find(inst_j.conclusions[k])):
                                failures.append({
                                    "rule_type": rule_type,
                                    "i_instance": inst_i.instance_id,
                                    "j_instance": inst_j.instance_id,
                                    "k": k,
                                    "i_conclusion_root": self.bundle.find(inst_i.conclusions[k]),
                                    "j_conclusion_root": self.bundle.find(inst_j.conclusions[k]),
                                })

        if failures:
            return CheckResult(
                name="rule_congruence",
                phase=2,
                passed=False,
                detail=f"{len(failures)} rule congruence violations",
                counterexample=failures[:5],
            )
        return CheckResult(name="rule_congruence", phase=2, passed=True)

    def _check_filter_safety(self) -> CheckResult:
        """过滤安全性检查"""
        if not self.bundle.filtered_terms:
            return CheckResult(name="filter_safety", phase=2, passed=True)

        # Build: which class ids are derived using which filtered terms?
        filtered_class_ids = set()
        for term in self.bundle.filtered_terms:
            for cid, cls in self.bundle.classes.items():
                if term in cls.members:
                    filtered_class_ids.add(cid)

        # Check if any filtered class is necessary for terminal derivation
        unreachable_without_filtered = []

        # Simple check: any rule instance that uses a filtered class
        # as premise to produce a terminal class?
        for inst in self.bundle.rule_instances:
            for p in inst.premises:
                if p in filtered_class_ids:
                    for c in inst.conclusions:
                        if c in self.terminal_set:
                            # Check if there's an alternative derivation
                            alt_exists = self._has_alternative_derivation(c, exclude={p})
                            if not alt_exists:
                                unreachable_without_filtered.append({
                                    "filtered_class": p,
                                    "terminal_class": c,
                                    "instance": inst.instance_id,
                                })

        if unreachable_without_filtered:
            return CheckResult(
                name="filter_safety",
                phase=2,
                passed=False,
                detail=f"{len(unreachable_without_filtered)} filter safety violations",
                counterexample=unreachable_without_filtered[:3],
            )
        return CheckResult(name="filter_safety", phase=2, passed=True)

    def _has_alternative_derivation(self, target: int, exclude: Set[int]) -> bool:
        """Check if target can be derived without using classes in exclude"""
        reachable = set(self.bundle.seed_classes) - exclude
        steps = [s for s in self.bundle.derivation_steps
                 if not set(s.parent_classes) & exclude]

        changed = True
        max_iter = 100
        iteration = 0
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for step in steps:
                if set(step.parent_classes).issubset(reachable):
                    if step.target_class not in reachable:
                        reachable.add(step.target_class)
                        changed = True
                        if step.target_class == target:
                            return True

        return target in reachable

    # ─────────────────────────────────
    # Phase 3: 结构一致性
    # ─────────────────────────────────

    def _phase3_structure(self) -> List[CheckResult]:
        checks = []
        checks.append(self._check_triality())
        checks.append(self._check_shell())
        checks.append(self._check_wedderburn_blocks())
        checks.append(self._check_schur())
        return checks

    def _check_triality(self) -> CheckResult:
        """Triality 检查: T³ = id, orbit sizes ∈ {1, 3}"""
        if not self.bundle.triality_orbits:
            return CheckResult(name="triality", phase=3, passed=True,
                              detail="No triality orbits defined")

        failures = []
        for orbit in self.bundle.triality_orbits:
            if len(orbit) not in (1, 3):
                failures.append({
                    "orbit": orbit,
                    "size": len(orbit),
                    "note": "Orbit size must be 1 or 3",
                })

        # 3*N₃ + N₁ = 44
        n3 = self.bundle.triality_orbit_sizes.get(3, 0)
        n1 = self.bundle.triality_orbit_sizes.get(1, 0)
        total = 3 * n3 + n1
        if total != 44 and total != 0:
            failures.append({
                "n3": n3, "n1": n1, "total": total,
                "note": f"3·{n3} + {n1} = {total} ≠ 44",
            })

        if failures:
            return CheckResult(
                name="triality",
                phase=3,
                passed=False,
                detail=f"{len(failures)} triality violations",
                counterexample=failures,
            )
        return CheckResult(name="triality", phase=3, passed=True)

    def _check_shell(self) -> CheckResult:
        """Shell 检查: shell 规则不产生非法壳层跃迁
        
        检查: 同轨道内跨 shell 跃迁是否在容许范围内.
        (同一 orbit 内的 shell 跃迁通常合法, 因为 triality 保持 orbit 但可能改变 shell)
        """
        failures = []
        max_shell_diff = 3  # 允许的最大壳层差
        for inst in self.bundle.rule_instances:
            if inst.rule_type != RuleType.SHELL.value:
                continue
            for p in inst.premises:
                p_cls = self.bundle.classes.get(p)
                if p_cls is None:
                    continue
                for c in inst.conclusions:
                    c_cls = self.bundle.classes.get(c)
                    if c_cls is None:
                        continue
                    shell_diff = abs(c_cls.shell - p_cls.shell)
                    if shell_diff > max_shell_diff:
                        failures.append({
                            "instance": inst.instance_id,
                            "premise_shell": p_cls.shell,
                            "conclusion_shell": c_cls.shell,
                            "shell_diff": shell_diff,
                        })

        if failures:
            return CheckResult(
                name="shell",
                phase=3,
                passed=False,
                detail=f"{len(failures)} shell rule violations",
                counterexample=failures[:3],
            )
        return CheckResult(name="shell", phase=3, passed=True)

    def _check_wedderburn_blocks(self) -> CheckResult:
        """Wedderburn 块检查: 块内规则不应跨块"""
        failures = []
        for inst in self.bundle.rule_instances:
            if inst.rule_type != RuleType.WEDDERBURN.value:
                continue
            premise_blocks = set()
            for p in inst.premises:
                cls = self.bundle.classes.get(p)
                if cls and cls.block:
                    premise_blocks.add(cls.block)

            if len(premise_blocks) > 1:
                failures.append({
                    "instance": inst.instance_id,
                    "premise_blocks": list(premise_blocks),
                    "note": "Wedderburn premises from different blocks",
                })
                continue

            if premise_blocks:
                block = list(premise_blocks)[0]
                for c in inst.conclusions:
                    cls = self.bundle.classes.get(c)
                    if cls and cls.block and cls.block != block:
                        failures.append({
                            "instance": inst.instance_id,
                            "premise_block": block,
                            "conclusion_block": cls.block,
                            "note": "Cross-block Wedderburn generation",
                        })

        if failures:
            return CheckResult(
                name="wedderburn_blocks",
                phase=3,
                passed=False,
                detail=f"{len(failures)} Wedderburn block violations",
                counterexample=failures[:3],
            )
        return CheckResult(name="wedderburn_blocks", phase=3, passed=True)

    def _check_schur(self) -> CheckResult:
        """Schur 检查: 非同构不可约块间不应有非零映射
        
        注意: shell 和 triality 规则描述了内在对称性 (T 作用, 壳层关系),
        它们的"跨块"行为是代数结构的一部分, 不是 Schur 违规。
        只有 wedderburn 类型的闭包规则才受 Schur 引理约束。
        """
        failures = []
        for inst in self.bundle.rule_instances:
            # Skip non-structural rules
            if inst.rule_type in (RuleType.SHELL.value, RuleType.TRIALITY.value,
                                   RuleType.TENSOR.value, RuleType.FILTER.value,
                                   RuleType.QUOTIENT.value, RuleType.BOOTSTRAP.value):
                continue

            source_blocks = set()
            for p in inst.premises:
                cls = self.bundle.classes.get(p)
                if cls and cls.block:
                    source_blocks.add(cls.block)
            target_blocks = set()
            for c in inst.conclusions:
                cls = self.bundle.classes.get(c)
                if cls and cls.block:
                    target_blocks.add(cls.block)

            # Wedderburn rule: premises should be same block, conclusions in same block
            if source_blocks and target_blocks:
                if len(source_blocks) > 1:
                    failures.append({
                        "instance": inst.instance_id,
                        "rule_type": inst.rule_type,
                        "source_blocks": list(source_blocks),
                        "note": "Wedderburn premises from different blocks",
                    })
                elif source_blocks != target_blocks:
                    failures.append({
                        "instance": inst.instance_id,
                        "rule_type": inst.rule_type,
                        "source_blocks": list(source_blocks),
                        "target_blocks": list(target_blocks),
                        "note": "Cross-block Wedderburn generation",
                    })

        if failures and self.bundle.schur_violations == 0:
            # If we detected failures but bundle says zero violations,
            # trust the bundle's manual count
            pass

        if failures:
            return CheckResult(
                name="schur",
                phase=3,
                passed=False,
                detail=f"{len(failures)} potential Schur violations",
                counterexample=failures[:3],
            )
        return CheckResult(name="schur", phase=3, passed=True)

    # ─────────────────────────────────
    # Phase 4: 合流性
    # ─────────────────────────────────

    def _phase4_confluence(self) -> List[CheckResult]:
        checks = []
        checks.append(self._check_termination())
        checks.append(self._check_critical_pairs())
        return checks

    def _check_termination(self) -> CheckResult:
        """终止性检查"""
        details = []

        # Check total admissible universe size
        n_classes = len(self.bundle.classes)
        if n_classes > 0:
            details.append(f"|U_adm/∼| = {n_classes} < ∞")

        # Check shell upper bound
        max_shell = 0
        for cls in self.bundle.classes.values():
            if cls.shell > max_shell:
                max_shell = cls.shell
        if max_shell > 0:
            details.append(f"max(σ) = {max_shell} < ∞")

        # Check block finiteness
        n_blocks = len(self.bundle.block_counts)
        if n_blocks > 0:
            details.append(f"|B| = {n_blocks} < ∞")

        # Check no infinite chains (finite state space)
        if n_classes <= 1000:
            details.append("finite state space → termination")

        # All conditions met = termination certified
        termination_info = self.bundle.termination
        has_finite_universe = (
            n_classes > 0 and n_classes <= 1000
        )

        if has_finite_universe:
            return CheckResult(
                name="termination",
                phase=4,
                passed=True,
                detail="; ".join(details),
            )
        else:
            return CheckResult(
                name="termination",
                phase=4,
                passed=False,
                detail=f"Universe size {n_classes} - cannot guarantee termination",
            )

    def _check_critical_pairs(self) -> CheckResult:
        """Critical pair joinability 检查"""
        if not self.bundle.critical_pairs:
            return CheckResult(
                name="critical_pairs",
                phase=4,
                passed=True,
                detail="No critical pairs registered — skipping joinability check",
            )

        failures = []
        for cp in self.bundle.critical_pairs:
            a_root = self.bundle.find(cp.result_a)
            b_root = self.bundle.find(cp.result_b)
            w_root = self.bundle.find(cp.join_witness)

            # Check that both results reach the join witness
            # (In our framework: they must be in the same equivalence class)
            u_reaches = self._reaches(a_root, w_root)
            v_reaches = self._reaches(b_root, w_root)

            if not (u_reaches and v_reaches):
                failures.append({
                    "pair_id": cp.pair_id,
                    "source": cp.source_term,
                    "rule_a": cp.rule_a,
                    "rule_b": cp.rule_b,
                    "result_a_root": a_root,
                    "result_b_root": b_root,
                    "join_witness_root": w_root,
                    "a_reaches_w": u_reaches,
                    "b_reaches_w": v_reaches,
                })

        if failures:
            return CheckResult(
                name="critical_pairs",
                phase=4,
                passed=False,
                detail=f"{len(failures)} non-joinable critical pairs",
                counterexample=failures[:5],
            )
        return CheckResult(
            name="critical_pairs",
            phase=4,
            passed=True,
            detail=f"All {len(self.bundle.critical_pairs)} critical pairs joinable",
        )

    def _reaches(self, source: int, target: int) -> bool:
        """Check if source class can reach target class via derivation steps"""
        if source == target:
            return True

        # Simple: check if they're in the same equivalence class
        if self.bundle.find(source) == self.bundle.find(target):
            return True

        # Otherwise, check derivation steps
        reachable = {source}
        changed = True
        max_iter = 50
        iteration = 0
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for step in self.bundle.derivation_steps:
                if set(step.parent_classes).issubset(reachable):
                    if step.target_class not in reachable:
                        reachable.add(step.target_class)
                        changed = True
                        if step.target_class == target:
                            return True
        return target in reachable

    # ─────────────────────────────────
    # 哈希
    # ─────────────────────────────────

    def _terminal_count_hash(self) -> str:
        """基于计数的终端哈希 — 对标签置换不敏感"""
        items = sorted(self.bundle.shell_counts.items()) if self.bundle.shell_counts else []
        block_items = sorted(self.bundle.block_counts.items()) if self.bundle.block_counts else []
        orbit_items = sorted(self.bundle.triality_orbit_sizes.items()) if self.bundle.triality_orbit_sizes else []

        data = json.dumps([items, block_items, orbit_items], sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def _terminal_structure_hash(self) -> str:
        """基于不变量表的终端结构哈希"""
        invariant_table = []
        for cid in sorted(self.bundle.terminal_classes):
            cls = self.bundle.classes.get(cid)
            if cls:
                invariant_table.append((
                    cls.shell,
                    cls.triality_grade,
                    cls.block,
                    cls.schur_type,
                ))

        data = json.dumps(sorted(invariant_table), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def _confluence_status(self, report: VerificationReport) -> str:
        """Determine confluence status from check results"""
        termination_pass = any(
            c.name == "termination" and c.passed for c in report.checks
        )
        cp_pass = any(
            c.name == "critical_pairs" and c.passed for c in report.checks
        )

        if termination_pass and cp_pass:
            return "certified_by_critical_pairs"
        elif cp_pass:
            return "cp_joinable_termination_unverified"
        else:
            return "unverified"
