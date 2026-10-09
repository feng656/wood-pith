#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from racpith.evaluation.report import build_markdown_report
from racpith.provenance import (
    atomic_write_json,
    atomic_write_text,
    read_json_object,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Markdown report from saved metrics and figures")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--figures", required=True)
    parser.add_argument("--contribution-metrics")
    parser.add_argument("--contribution-figures")
    parser.add_argument("--baseline-metrics")
    parser.add_argument(
        "--audit",
        action="append",
        required=True,
        help=(
            "PASS racpith.combined_audit.v1 artifact to bind into the sealed "
            "report; provide exactly three scopes (repeatable)"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.audit) != 3:
        raise ValueError(
            "exactly three --audit inputs are required: main, partition, and target-domain"
        )
    output = Path(args.output)
    registered_inputs = {
        "localization_denominator": Path(args.metrics) / "denominators.json",
        "visualization_index": Path(args.figures) / "visualization_index.json",
    }
    optional_inputs = {
        "runtime_denominator": Path(args.metrics) / "runtime_denominator.json",
        "contribution_evaluation_index": (
            Path(args.contribution_metrics) / "contribution_evaluation_index.json"
            if args.contribution_metrics
            else None
        ),
        "contribution_visualization_index": (
            Path(args.contribution_figures) / "contribution_visualization_index.json"
            if args.contribution_figures
            else None
        ),
        "baseline_denominator": (
            Path(args.baseline_metrics) / "baseline_denominator.json"
            if args.baseline_metrics
            else None
        ),
    }
    registered_inputs.update(
        {
            name: path
            for name, path in optional_inputs.items()
            if path is not None
        }
    )
    audit_summaries = []
    seen_audit_paths: set[Path] = set()
    seen_audit_hashes: set[str] = set()
    target_audit_count = 0
    for ordinal, audit_value in enumerate(args.audit):
        audit_path = Path(audit_value).expanduser().resolve()
        if audit_path in seen_audit_paths:
            raise ValueError(f"duplicate report audit input: {audit_path}")
        seen_audit_paths.add(audit_path)
        audit_sha256 = sha256_file(audit_path)
        if audit_sha256 in seen_audit_hashes:
            raise ValueError(f"duplicate audit content supplied under another path: {audit_path}")
        seen_audit_hashes.add(audit_sha256)
        audit = read_json_object(audit_path)
        if audit.get("schema_version") != "racpith.combined_audit.v1":
            raise ValueError(f"unsupported report audit schema: {audit_path}")
        if audit.get("status") != "PASS":
            raise ValueError(f"refusing to seal report with a failed audit: {audit_path}")
        artifact_audit = audit.get("artifact_audit")
        if (
            not isinstance(artifact_audit, dict)
            or artifact_audit.get("schema_version") != "racpith.audit.v1"
            or artifact_audit.get("status") != "PASS"
        ):
            raise ValueError(
                f"combined audit lacks a PASS artifact audit: {audit_path}"
            )
        static_findings = audit.get("static_findings")
        artifact_findings = artifact_audit.get("findings")
        if not isinstance(static_findings, list):
            raise ValueError(f"combined audit has invalid static findings: {audit_path}")
        if not isinstance(artifact_findings, list):
            raise ValueError(f"combined audit has invalid artifact findings: {audit_path}")
        for finding in [*static_findings, *artifact_findings]:
            if (
                not isinstance(finding, dict)
                or finding.get("severity") not in {"ERROR", "WARNING"}
            ):
                raise ValueError(
                    f"combined audit contains a malformed finding: {audit_path}"
                )
        static_errors = sum(
            1
            for finding in static_findings
            if isinstance(finding, dict) and finding.get("severity") == "ERROR"
        )
        artifact_errors = sum(
            1
            for finding in artifact_findings
            if isinstance(finding, dict) and finding.get("severity") == "ERROR"
        )
        if static_errors or artifact_errors:
            raise ValueError(
                f"PASS audit contains ERROR findings and cannot seal a report: "
                f"{audit_path}"
            )
        denominator = artifact_audit.get("denominator")
        if not isinstance(denominator, dict):
            raise ValueError(f"combined audit lacks an artifact denominator: {audit_path}")
        try:
            target_records = int(denominator.get("target_domain_records", 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"combined audit has an invalid target-domain denominator: {audit_path}"
            ) from exc
        if target_records < 0:
            raise ValueError(f"combined audit has a negative target-domain denominator: {audit_path}")
        is_target_audit = target_records > 0
        if is_target_audit:
            target_audit_count += 1
            if not denominator.get("target_domain_index_sha256"):
                raise ValueError(
                    f"target-domain audit lacks its index hash: {audit_path}"
                )
        audit_summaries.append(
            {
                "name": audit_path.name,
                "scope": "target_domain" if is_target_audit else "localization_or_contribution",
                "sha256": audit_sha256,
                "static_findings": len(static_findings),
                "artifact_findings": len(artifact_findings),
                "target_domain_records": target_records,
            }
        )
        registered_inputs[f"audit_{ordinal:02d}"] = audit_path
    if target_audit_count != 1:
        raise ValueError(
            "exactly one of the three audits must cover a non-empty target-domain index"
        )
    for name, path in registered_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"report input index is absent: {name}={path}")
    index_output = output.with_suffix(output.suffix + ".index.json")
    reserved_outputs = {output.resolve(), index_output.resolve()}
    collisions = {
        name: path
        for name, path in registered_inputs.items()
        if path.resolve() in reserved_outputs
    }
    if collisions:
        raise ValueError(f"report output collides with an input index: {collisions}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_markdown_report(
        args.metrics,
        args.figures,
        args.contribution_metrics,
        args.contribution_figures,
        args.baseline_metrics,
    )
    if audit_summaries:
        audit_lines = [
            "## 11. 封存审计",
            "",
            "以下 `racpith.combined_audit.v1` 均为 PASS；其文件内容哈希由报告 sidecar 绑定：",
            "",
        ]
        audit_lines.extend(
            "- `{name}`（{scope}）：static findings={static_findings}，artifact findings={artifact_findings}，target-domain records={target_domain_records}".format(
                **summary
            )
            for summary in audit_summaries
        )
        report = report.rstrip() + "\n\n" + "\n".join(audit_lines) + "\n"
    atomic_write_text(output, report)
    input_hashes = {
        name: sha256_file(path) for name, path in registered_inputs.items()
    }
    atomic_write_json(
        index_output,
        {
            "schema_version": "racpith.report_index.v1",
            "audit_status": "PASS",
            "audit_count": len(args.audit),
            "audit_summaries": audit_summaries,
            "report_path": output.name,
            "report_sha256": sha256_file(output),
            "input_artifact_hashes": input_hashes,
            "input_artifacts": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": input_hashes[name],
                }
                for name, path in registered_inputs.items()
            },
        },
    )
    print(f"report: {output}")


if __name__ == "__main__":
    main()
