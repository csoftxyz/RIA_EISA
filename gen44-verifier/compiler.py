"""
gen44 Terminal Certificate Verifier v0.9 — 证明编译管线 (P0 fixed)

将 gen44 程序运行日志编译为可验证的 ProofBundle

P0 修复:
  - seed 命令显式声明
  - bootstrap 规则类型
  - merge 后 canonicalize 等价类根
  - nf 映射到 canonical representative
  - member 级别属性记录
"""

from models import (
    ProofBundle, RawTerm, EquivalenceClass, RuleInstance,
    DerivationStep, MergeEvent, CriticalPair, RuleType
)
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import re


class ProofCompiler:
    """证明编译器 — 将程序日志转换为数学证书"""

    def __init__(self):
        self.bundle = ProofBundle()
        self._next_class_id = 1
        self._raw_terms: Dict[str, RawTerm] = {}
        self._expr_to_class: Dict[str, int] = {}
        self._explicit_seeds: List[str] = []
        self._terminal_size: Optional[int] = None
        # P0 fix: collect all fire events separately from raw_terms
        self._fire_events: List[Tuple[str, List[str], List[str]]] = []

    def compile_from_log(self, log_lines: List[str]) -> ProofBundle:
        """
        从 gen44 程序日志编译证明包。

        日志格式:
            add <id> [shell=<n>] [grade=<g>] [block=<b>] [schur=<s>]
            fire triality <x> -> <y>,<z>
            fire shell <x> -> <y>
            fire tensor (<x>,<y>,<z>) -> <w>
            fire wedderburn <x>,<y> -> <z>
            fire bootstrap <x> -> <y>
            merge <x> <y>
            filter <x>
            seed <id>
            terminal size <n>
        """
        self._next_class_id = 1
        self._raw_terms = {}
        self._expr_to_class = {}
        self._explicit_seeds = []
        self._terminal_size = None

        for line in log_lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            self._parse_line(line)

        self._build_universe()
        self._build_derivation_steps()
        self._build_invariants()
        return self.bundle

    def _parse_line(self, line: str):
        parts = line.split()

        if parts[0] == "add":
            self._parse_add(parts[1:])
        elif parts[0] == "fire":
            self._parse_fire(parts[1:])
        elif parts[0] == "merge":
            self._parse_merge(parts[1:])
        elif parts[0] == "filter":
            self._parse_filter(parts[1:])
        elif parts[0] == "seed":
            self._explicit_seeds.append(parts[1])
        elif parts[0] == "terminal" and parts[1] == "size":
            self._terminal_size = int(parts[2])

    def _parse_add(self, parts: List[str]):
        term_id = parts[0]
        attrs = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                attrs[k] = v

        term = RawTerm(
            id=term_id,
            expr=term_id,
            shell=int(attrs.get("shell", 0)),
            triality_grade=int(attrs.get("grade", 0)),
            block=attrs.get("block", ""),
            schur_type=attrs.get("schur", "admissible"),
        )
        self._raw_terms[term_id] = term

    def _parse_fire(self, parts: List[str]):
        """P0 fix: collect ALL fire events (not just first-parent-wins)"""
        rule_type = parts[0]

        arrow_idx = None
        for i, p in enumerate(parts):
            if p == "->":
                arrow_idx = i
                break

        if arrow_idx is None:
            return

        premises_str = parts[1:arrow_idx]
        conclusions_str = parts[arrow_idx + 1:]

        premises = []
        prem_text = " ".join(premises_str)

        if rule_type == RuleType.TENSOR.value:
            m = re.search(r'\((.+)\)', prem_text)
            if m:
                premises = [s.strip() for s in m.group(1).split(",")]
        elif rule_type in (RuleType.TRIALITY.value, RuleType.BOOTSTRAP.value):
            premises = [premises_str[0]]
        elif rule_type == RuleType.WEDDERBURN.value:
            premises = [s.strip() for s in prem_text.split(",")]
        else:
            premises = [premises_str[0]]

        conclusions = [s.strip().rstrip(",") for s in conclusions_str]

        # Ensure terms exist in raw_terms dict
        for p in premises:
            if p not in self._raw_terms:
                self._raw_terms[p] = RawTerm(id=p, expr=p)
        for c in conclusions:
            if c not in self._raw_terms:
                self._raw_terms[c] = RawTerm(id=c, expr=c)

        # P0 fix: Record as fire event (allows multiple rules per target)
        self._fire_events.append((rule_type, list(premises), list(conclusions)))

        # Backward compat: first-parent-wins for seed auto-detection
        for c in conclusions:
            t = self._raw_terms[c]
            if not t.raw_parents:
                t.raw_parents = premises
                t.rule_id = f"fire-{rule_type}"

    def _parse_merge(self, parts: List[str]):
        if len(parts) >= 2:
            self.bundle.merge_events.append(MergeEvent(left=parts[0], right=parts[1]))

    def _parse_filter(self, parts: List[str]):
        if parts:
            self.bundle.filtered_terms.append(parts[0])

    # ── P0 修复: _build_universe ──

    def _build_universe(self):
        """从原始项构建等价类和终端表 (P0 canonicalized)"""

        # Step 1: Assign class ids
        for term_id in self._raw_terms:
            cid = self._next_class_id
            self._next_class_id += 1
            self._expr_to_class[term_id] = cid
            self.bundle.parent[cid] = cid

            term = self._raw_terms[term_id]
            eq_class = EquivalenceClass(
                class_id=cid,
                canonical_rep=term_id,
                members=[term_id],
                shell=term.shell,
                triality_grade=term.triality_grade,
                block=term.block,
                schur_type=term.schur_type,
            )
            self.bundle.classes[cid] = eq_class

        # Step 2: Apply merges
        for merge in self.bundle.merge_events:
            if merge.left in self._expr_to_class and merge.right in self._expr_to_class:
                c1 = self._expr_to_class[merge.left]
                c2 = self._expr_to_class[merge.right]
                self.bundle.union(c1, c2)

        # Step 3: Canonicalize _expr_to_class to roots (P0 fix)
        self._expr_to_class = {
            term: self.bundle.find(cid)
            for term, cid in self._expr_to_class.items()
        }

        # Step 4: Group members by equivalence class root
        root_to_members: Dict[int, List[str]] = defaultdict(list)
        for term_id, root in self._expr_to_class.items():
            root_to_members[root].append(term_id)

        # Step 5: Build terminal classes (only roots, P0 fix)
        new_classes: Dict[int, EquivalenceClass] = {}
        terminal_classes = []
        seen_roots = set()
        for root in sorted(root_to_members.keys()):
            if root in seen_roots:
                continue
            seen_roots.add(root)

            members = root_to_members[root]
            canonical = min(members, key=lambda m: (len(m), m))

            # Build member-level attributes (P0 fix)
            member_attrs = {}
            shell = 0
            grade = 0
            block = ""
            schur = "admissible"
            for m in members:
                if m in self._raw_terms:
                    t = self._raw_terms[m]
                    member_attrs[m] = {
                        "shell": t.shell,
                        "triality_grade": t.triality_grade,
                        "block": t.block,
                        "schur_type": t.schur_type,
                    }
                    if t.shell:
                        shell = t.shell
                    if t.triality_grade:
                        grade = t.triality_grade
                    if t.block:
                        block = t.block
                    if t.schur_type:
                        schur = t.schur_type

            eq_class = EquivalenceClass(
                class_id=root,
                canonical_rep=canonical,
                members=members,
                shell=shell,
                triality_grade=grade,
                block=block,
                schur_type=schur,
            )
            new_classes[root] = eq_class
            terminal_classes.append(root)

            # P0 fix: normal_forms → canonical representative
            for m in members:
                self.bundle.normal_forms[m] = canonical

        # Replace classes dict with only roots (P0 fix)
        self.bundle.classes = new_classes
        self.bundle.terminal_classes = terminal_classes

        # Step 6: Build rule instances from raw terms
        self._build_rule_instances()

        # P0 fix: Copy raw_terms to bundle for attribute_consistency check
        self.bundle.raw_terms = dict(self._raw_terms)

        # Step 7: Identify seed classes
        if self._explicit_seeds:
            seed_terms = set(self._explicit_seeds)
        else:
            seed_terms = set()
            for term_id, term in self._raw_terms.items():
                if not term.raw_parents:
                    seed_terms.add(term_id)

        # Map seed term ids to class roots
        seed_class_roots = set()
        for t in seed_terms:
            if t in self._expr_to_class:
                seed_class_roots.add(self._expr_to_class[t])

        self.bundle.seed_classes = sorted(seed_class_roots)

    def _build_rule_instances(self):
        """P0 fix: 从 _fire_events 构建规则实例 (支持多规则同一目标)"""
        inst_count = 0
        for rule_type, premises, conclusions in self._fire_events:
            parent_roots = []
            for p in premises:
                if p in self._expr_to_class:
                    parent_roots.append(self._expr_to_class[p])

            conclusion_roots = []
            for c in conclusions:
                if c in self._expr_to_class:
                    conclusion_roots.append(self._expr_to_class[c])

            if not parent_roots or not conclusion_roots:
                continue

            # Map rule type string to RuleType value
            if rule_type == "bootstrap":
                rtype = RuleType.BOOTSTRAP.value
            elif rule_type == "triality":
                rtype = RuleType.TRIALITY.value
            elif rule_type == "shell":
                rtype = RuleType.SHELL.value
            elif rule_type == "tensor":
                rtype = RuleType.TENSOR.value
            elif rule_type == "wedderburn":
                rtype = RuleType.WEDDERBURN.value
            else:
                rtype = rule_type

            inst = RuleInstance(
                instance_id=f"INST-{inst_count:04d}",
                rule_type=rtype,
                premises=parent_roots,
                conclusions=conclusion_roots,
                side_conditions={"fire_event_index": inst_count},
            )
            self.bundle.rule_instances.append(inst)
            inst_count += 1

    def _build_derivation_steps(self):
        """构建推导森林 (P0: 使用 terminal root ids)"""
        depth_map: Dict[int, int] = {}
        for s in self.bundle.seed_classes:
            depth_map[s] = 0

        # BFS to find derivations
        max_iter = 200
        changed = True
        iteration = 0
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for inst in self.bundle.rule_instances:
                parents_with_depth = [
                    p for p in inst.premises if p in depth_map
                ]
                if len(parents_with_depth) == len(inst.premises):
                    new_depth = max(depth_map[p] for p in inst.premises) + 1
                    for c in inst.conclusions:
                        if c not in depth_map or depth_map[c] > new_depth:
                            depth_map[c] = new_depth
                            changed = True

        # Build one derivation step per target
        rule_by_target: Dict[int, List[RuleInstance]] = defaultdict(list)
        for inst in self.bundle.rule_instances:
            for c in inst.conclusions:
                rule_by_target[c].append(inst)

        for target, depth in depth_map.items():
            if depth == 0:
                ds = DerivationStep(
                    target_class=target,
                    rule_instance_id="seed",
                    parent_classes=[],
                    depth=0,
                )
                self.bundle.derivation_steps.append(ds)
                continue

            # P0 fix: prefer rule with reachable parents (lowest max depth)
            best_inst = None
            best_max_depth = 999
            for inst in rule_by_target.get(target, []):
                if all(p in depth_map for p in inst.premises):
                    max_pd = max(depth_map[p] for p in inst.premises)
                    if max_pd < best_max_depth:
                        best_max_depth = max_pd
                        best_inst = inst

            if best_inst:
                ds = DerivationStep(
                    target_class=target,
                    rule_instance_id=best_inst.instance_id,
                    parent_classes=list(best_inst.premises),
                    depth=depth,
                )
                self.bundle.derivation_steps.append(ds)

    def _build_invariants(self):
        shell_counts: Dict[int, int] = defaultdict(int)
        for cid in self.bundle.terminal_classes:
            cls = self.bundle.classes.get(cid)
            if cls:
                shell_counts[cls.shell] += 1
        self.bundle.shell_counts = dict(shell_counts)

        block_counts: Dict[str, int] = defaultdict(int)
        for cid in self.bundle.terminal_classes:
            cls = self.bundle.classes.get(cid)
            if cls and cls.block:
                block_counts[cls.block] += 1
        self.bundle.block_counts = dict(block_counts)

        orbit_sizes: Dict[int, int] = defaultdict(int)
        for orbit in self.bundle.triality_orbits:
            sz = len(orbit)
            if sz in (1, 3):
                orbit_sizes[sz] += 1
        self.bundle.triality_orbit_sizes = dict(orbit_sizes)

        self.bundle.invariants = {
            "total_classes": len(self.bundle.terminal_classes),
            "num_seeds": len(self.bundle.seed_classes),
            "num_rule_instances": len(self.bundle.rule_instances),
            "num_filtered": len(self.bundle.filtered_terms),
            "num_merges": len(self.bundle.merge_events),
        }
