#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from racpith.audit.run import audit_run
from racpith.audit.static import audit_source_tree, count_errors
from racpith.config import load_config
from racpith.provenance import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Static and artifact-contract audit")
    parser.add_argument("--source-root", default="src/racpith")
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--result-index")
    parser.add_argument("--contributions")
    parser.add_argument("--contribution-index")
    parser.add_argument("--uncertainty-index")
    parser.add_argument("--target-domain-index")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=[])
    args = parser.parse_args()
    frozen = load_config(args.config)
    static_findings = audit_source_tree(args.source_root)
    script_root = Path(__file__).resolve().parent
    if script_root.resolve() != Path(args.source_root).expanduser().resolve():
        static_findings.extend(audit_source_tree(script_root))
    artifact = audit_run(
        args.crop_manifest,
        args.evidence_index,
        args.predictions,
        frozen.sha256,
        contribution_root=args.contributions,
        contribution_index_path=args.contribution_index,
        splits=set(args.split) if args.split else None,
        uncertainty_index_path=args.uncertainty_index,
        result_index_path=args.result_index,
        target_domain_index_path=args.target_domain_index,
    )
    report = {
        "schema_version": "racpith.combined_audit.v1",
        "static_findings": static_findings,
        "artifact_audit": artifact,
        "status": "FAIL"
        if count_errors(static_findings) or artifact["status"] == "FAIL"
        else "PASS",
    }
    output = Path(args.output)
    atomic_write_json(output, report)
    print(f"audit status: {report['status']}; report={output}")
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
