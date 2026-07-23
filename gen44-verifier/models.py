"""
gen44 Terminal Certificate Verifier v0.9 — 核心数据结构
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import json


class RuleType(Enum):
    TRIALITY = "triality"
    SHELL = "shell"
    TENSOR = "tensor"
    WEDDERBURN = "wedderburn"
    QUOTIENT = "quotient"
    FILTER = "filter"
    BOOTSTRAP = "bootstrap"


class SchurType(Enum):
    ADMISSIBLE = "admissible"
    FORBIDDEN = "forbidden"
    SCALAR = "scalar"


@dataclass
class RawTerm:
    """原始项 — 来自程序日志"""
    id: str
    expr: str
    shell: int = 0
    triality_grade: int = 0
    block: str = ""
    schur_type: str = "admissible"
    raw_parents: List[str] = field(default_factory=list)
    rule_id: str = ""


@dataclass
class EquivalenceClass:
    """等价类 — 商化后的终端对象"""
    class_id: int
    canonical_rep: str
    members: List[str] = field(default_factory=list)
    shell: int = 0
    triality_orbit: int = 0
    triality_grade: int = 0
    block: str = ""
    schur_type: str = "admissible"


@dataclass
class RuleInstance:
    """规则实例 — 一次规则的触发"""
    instance_id: str
    rule_type: str
    premises: List[int]          # class ids (终端的等价类)
    conclusions: List[int]       # class ids
    side_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeEvent:
    """合并事件 — 两个原始项被声明为等价"""
    left: str
    right: str


@dataclass
class DerivationStep:
    """推导步骤 — 从前提类到达目标类"""
    target_class: int
    rule_instance_id: str
    parent_classes: List[int]
    depth: int = 0


@dataclass
class CriticalPair:
    """Critical pair — 合流性检查的核心"""
    pair_id: str
    source_term: str
    rule_a: str
    result_a: int          # class id
    rule_b: str
    result_b: int          # class id
    join_witness: int      # class id (if exists)


@dataclass
class ProofBundle:
    """证明包 — 完整的可验证证书"""
    # 基本信息
    bundle_id: str = "gen44-v0.9"
    description: str = ""

    # Seed
    seed_classes: List[int] = field(default_factory=list)

    # 终端等价类
    terminal_classes: List[int] = field(default_factory=list)
    classes: Dict[int, EquivalenceClass] = field(default_factory=dict)

    # 原始项 (用于规范化检查)
    raw_terms: Dict[str, RawTerm] = field(default_factory=dict)

    # 规范形
    normal_forms: Dict[str, str] = field(default_factory=dict)

    # 等价关系 (union-find)
    parent: Dict[int, int] = field(default_factory=dict)

    # 合并事件
    merge_events: List[MergeEvent] = field(default_factory=list)

    # 规则实例
    rule_instances: List[RuleInstance] = field(default_factory=list)

    # 推导森林
    derivation_steps: List[DerivationStep] = field(default_factory=list)

    # 关键对
    critical_pairs: List[CriticalPair] = field(default_factory=list)

    # 被过滤项
    filtered_terms: List[str] = field(default_factory=list)

    # 不变量
    invariants: Dict[str, Any] = field(default_factory=dict)
    shell_counts: Dict[int, int] = field(default_factory=dict)
    triality_orbit_sizes: Dict[int, int] = field(default_factory=dict)
    block_counts: Dict[str, int] = field(default_factory=dict)
    schur_violations: int = 0

    # 终止性证书
    termination: Dict[str, Any] = field(default_factory=dict)

    # Triality orbits
    triality_orbits: List[List[int]] = field(default_factory=list)

    # --- union-find helpers ---
    def find(self, x: int) -> int:
        """Union-find find with path compression"""
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        """Union-find union"""
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry

    def same_class(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def class_members(self, root: int) -> Set[int]:
        """Get all members of an equivalence class"""
        return {k for k in self.parent if self.find(k) == root}

    # --- serialization ---
    def to_dict(self) -> dict:
        result = {
            "bundle_id": self.bundle_id,
            "description": self.description,
            "seed_classes": self.seed_classes,
            "terminal_classes": self.terminal_classes,
            "classes": {
                str(k): {
                    "class_id": v.class_id,
                    "canonical_rep": v.canonical_rep,
                    "members": v.members,
                    "shell": v.shell,
                    "triality_orbit": v.triality_orbit,
                    "triality_grade": v.triality_grade,
                    "block": v.block,
                    "schur_type": v.schur_type,
                }
                for k, v in self.classes.items()
            },
            "normal_forms": self.normal_forms,
            "parent": {str(k): v for k, v in self.parent.items()},
            "merge_events": [{"left": m.left, "right": m.right} for m in self.merge_events],
            "rule_instances": [
                {
                    "instance_id": r.instance_id,
                    "rule_type": r.rule_type,
                    "premises": r.premises,
                    "conclusions": r.conclusions,
                    "side_conditions": r.side_conditions,
                }
                for r in self.rule_instances
            ],
            "derivation_steps": [
                {
                    "target_class": d.target_class,
                    "rule_instance_id": d.rule_instance_id,
                    "parent_classes": d.parent_classes,
                    "depth": d.depth,
                }
                for d in self.derivation_steps
            ],
            "critical_pairs": [
                {
                    "pair_id": cp.pair_id,
                    "source_term": cp.source_term,
                    "rule_a": cp.rule_a,
                    "result_a": cp.result_a,
                    "rule_b": cp.rule_b,
                    "result_b": cp.result_b,
                    "join_witness": cp.join_witness,
                }
                for cp in self.critical_pairs
            ],
            "filtered_terms": self.filtered_terms,
            "invariants": self.invariants,
            "shell_counts": self.shell_counts,
            "triality_orbit_sizes": self.triality_orbit_sizes,
            "block_counts": self.block_counts,
            "schur_violations": self.schur_violations,
            "termination": self.termination,
            "triality_orbits": self.triality_orbits,
        }
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "ProofBundle":
        bundle = cls(
            bundle_id=d.get("bundle_id", "gen44-v0.9"),
            description=d.get("description", ""),
            seed_classes=d.get("seed_classes", []),
            terminal_classes=d.get("terminal_classes", []),
            normal_forms=d.get("normal_forms", {}),
            parent={int(k): v for k, v in d.get("parent", {}).items()},
            merge_events=[
                MergeEvent(left=m["left"], right=m["right"])
                for m in d.get("merge_events", [])
            ],
            filtered_terms=d.get("filtered_terms", []),
            invariants=d.get("invariants", {}),
            shell_counts={int(k): v for k, v in d.get("shell_counts", {}).items()},
            triality_orbit_sizes={int(k): v for k, v in d.get("triality_orbit_sizes", {}).items()},
            block_counts=d.get("block_counts", {}),
            schur_violations=d.get("schur_violations", 0),
            termination=d.get("termination", {}),
            triality_orbits=d.get("triality_orbits", []),
        )

        for k, v in d.get("classes", {}).items():
            cid = int(k)
            bundle.classes[cid] = EquivalenceClass(
                class_id=v["class_id"],
                canonical_rep=v["canonical_rep"],
                members=v.get("members", []),
                shell=v.get("shell", 0),
                triality_orbit=v.get("triality_orbit", 0),
                triality_grade=v.get("triality_grade", 0),
                block=v.get("block", ""),
                schur_type=v.get("schur_type", "admissible"),
            )

        for r in d.get("rule_instances", []):
            bundle.rule_instances.append(RuleInstance(
                instance_id=r["instance_id"],
                rule_type=r["rule_type"],
                premises=r["premises"],
                conclusions=r["conclusions"],
                side_conditions=r.get("side_conditions", {}),
            ))

        for ds in d.get("derivation_steps", []):
            bundle.derivation_steps.append(DerivationStep(
                target_class=ds["target_class"],
                rule_instance_id=ds["rule_instance_id"],
                parent_classes=ds["parent_classes"],
                depth=ds.get("depth", 0),
            ))

        for cp in d.get("critical_pairs", []):
            bundle.critical_pairs.append(CriticalPair(
                pair_id=cp["pair_id"],
                source_term=cp["source_term"],
                rule_a=cp["rule_a"],
                result_a=cp["result_a"],
                rule_b=cp["rule_b"],
                result_b=cp["result_b"],
                join_witness=cp["join_witness"],
            ))

        return bundle

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ProofBundle":
        with open(path) as f:
            return cls.from_dict(json.load(f))
