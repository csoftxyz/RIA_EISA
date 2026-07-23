#!/usr/bin/env python3
"""
gen44 Terminal Certificate Verifier v0.9 — 主入口

用法:
  # 生成测试数据
  python3 main.py generate

  # 验证所有测试 bundle
  python3 main.py verify --all

  # 验证单个 bundle
  python3 main.py verify --bundle /path/to/bundle.json

  # 验证 perfect_44 并显示报告
  python3 main.py verify --perfect

  # 运行所有场景的全套测试
  python3 main.py test
"""

import sys
import os
import argparse
import json
from typing import Dict, List, Tuple

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ProofBundle
from verifier import TerminalCertificateVerifier, VerificationReport


def cmd_generate(args):
    """生成测试数据"""
    from generate_test_data import generate_all, ALL_SCENARIOS

    output_dir = args.output or "/tmp/gen44-test-bundles"
    print("=" * 64)
    print("  gen44 Test Data Generator")
    print("=" * 64)
    print(f"  Output: {output_dir}\n")

    results = generate_all(output_dir)

    print(f"\nGenerated {len(results)} test bundles:")
    for name, info in results.items():
        print(f"  {name:30s}  |R|={info['terminal_size']:3d}  "
              f"rules={info['num_rules']:4d}  CPs={info['num_critical_pairs']:3d}")


def cmd_verify(args):
    """验证证明包"""
    if args.all:
        # Verify all generated test bundles
        test_dir = args.test_dir or "/tmp/gen44-test-bundles"
        results = []
        for fname in sorted(os.listdir(test_dir)):
            if fname.endswith(".json"):
                path = os.path.join(test_dir, fname)
                name = fname.replace(".json", "")
                try:
                    bundle = ProofBundle.load(path)
                    verifier = TerminalCertificateVerifier(bundle)
                    report = verifier.verify()
                    results.append((name, report))
                    _print_summary(name, report)
                except Exception as e:
                    print(f"\n  ERROR loading {fname}: {e}")

        _print_totals(results)

    elif args.perfect:
        # Verify just the perfect_44 bundle
        from generate_test_data import perfect_44
        bundle = perfect_44()
        verifier = TerminalCertificateVerifier(bundle)
        report = verifier.verify()
        report.print_report()

    elif args.bundle:
        path = args.bundle
        bundle = ProofBundle.load(path)
        verifier = TerminalCertificateVerifier(bundle)
        report = verifier.verify()
        report.print_report()

        if args.output:
            report.save(args.output)
            print(f"\nReport saved to {args.output}")

    else:
        print("Specify --all, --perfect, or --bundle <path>")


def cmd_test(args):
    """运行完整测试套件并验证预期"""
    from generate_test_data import ALL_SCENARIOS

    # Expected results for each scenario
    EXPECTED = {
        "perfect_44": ("PASS", []),
        "undercount_43": ("FAIL", ["count_44"]),
        "duplicate_classes": ("FAIL", ["no_duplicates"]),
        "unreachable_class": ("FAIL", ["reachability"]),
        "closure_failure": ("FAIL", ["closure_mod_equiv"]),
        "triality_violation": ("FAIL", ["triality"]),
        "schur_violation": ("FAIL", ["schur"]),
        "congruence_violation": ("FAIL", ["rule_congruence"]),
        "confluence_certified": ("PASS", []),
        "gen44_sample": ("PASS", []),
    }

    passed = 0
    failed = 0
    results = []

    print("=" * 72)
    print("  gen44 Terminal Certificate Verifier v0.9 — Test Suite")
    print("=" * 72)

    for name in sorted(ALL_SCENARIOS.keys()):
        factory = ALL_SCENARIOS[name]
        bundle = factory()
        verifier = TerminalCertificateVerifier(bundle)
        report = verifier.verify()

        results.append((name, report))

        # Check against expectations
        expected = EXPECTED.get(name, None)
        if expected is None:
            # No expectations set (e.g., gen44_sample)
            print(f"\n  {name}: {report.status} (no expected result)")
            _print_failed_checks(report)
            continue

        expected_status, expected_fails = expected

        # Check status
        status_ok = (report.status == expected_status)

        # Check expected failures are present
        actual_fails = set(report.failed_checks)
        expected_fails_set = set(expected_fails)
        fails_ok = expected_fails_set.issubset(actual_fails) if expected_fails else True

        if status_ok and fails_ok:
            print(f"  ✅ {name:30s}  STATUS={report.status}  (expected)")
            passed += 1
        else:
            print(f"  ❌ {name:30s}  STATUS={report.status}  (expected {expected_status})")
            if not fails_ok:
                missing = expected_fails_set - actual_fails
                print(f"       Missing expected failures: {missing}")
                print(f"       Actual failures: {report.failed_checks}")
            failed += 1

        _print_failed_checks(report)

    # Print summary
    print("\n" + "=" * 72)
    print(f"  Results: {passed} passed, {failed} failed, {len(results)} total")
    print("=" * 72)

    # Print hash comparison for PASS bundles
    pass_results = [(n, r) for n, r in results if r.status == "PASS"]
    if len(pass_results) > 1:
        print(f"\n  All-PASS Bundles Hash Comparison:")
        unique_hashes = set()
        for n, r in pass_results:
            h = r.hash_count[:16]
            unique_hashes.add(h)
            print(f"    {n:30s}  hash={h}...")
        if len(unique_hashes) == 1:
            print(f"  ✅ All {len(pass_results)} PASS bundles have identical count hash")
        else:
            print(f"  ⚠️  {len(unique_hashes)} distinct hashes among {len(pass_results)} PASS bundles")

    return 0 if failed == 0 else 1


def _print_summary(name: str, report: VerificationReport):
    """打印单行摘要"""
    icon = "✅" if report.status == "PASS" else "❌"
    print(f"  {icon} {name:30s}  {report.status:4s}  "
          f"checks={sum(1 for c in report.checks if c.passed)}/{len(report.checks)}  "
          f"elapsed={report.elapsed_ms:.1f}ms")

    if report.failed_checks:
        print(f"       Failed: {', '.join(report.failed_checks)}")


def _print_failed_checks(report: VerificationReport):
    """打印失败详情"""
    for fc in report.failed_checks:
        for c in report.checks:
            if c.name == fc and c.counterexample:
                print(f"         └─ {c.name}: {str(c.counterexample)[:120]}")


def _print_totals(results: List[Tuple[str, VerificationReport]]):
    """打印汇总"""
    n_pass = sum(1 for _, r in results if r.status == "PASS")
    n_fail = sum(1 for _, r in results if r.status == "FAIL")
    print(f"\n{'=' * 64}")
    print(f"  Total: {len(results)} bundles | {n_pass} PASS | {n_fail} FAIL")
    print(f"{'=' * 64}")


def main():
    parser = argparse.ArgumentParser(
        description="gen44 Terminal Certificate Verifier v0.9"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate test data")
    gen_parser.add_argument("--output", "-o", help="Output directory")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Verify proof bundles")
    verify_parser.add_argument("--all", action="store_true", help="Verify all test bundles")
    verify_parser.add_argument("--perfect", action="store_true", help="Verify perfect_44")
    verify_parser.add_argument("--bundle", "-b", help="Path to bundle JSON")
    verify_parser.add_argument("--test-dir", "-d", help="Directory of test bundles")
    verify_parser.add_argument("--output", "-o", help="Save report to file")

    # test
    subparsers.add_parser("test", help="Run full test suite")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "test":
        sys.exit(cmd_test(args))
    else:
        parser.print_help()
        # Default: run test suite
        print("\nNo command specified — running test suite by default.\n")
        sys.exit(cmd_test(args))


if __name__ == "__main__":
    main()
