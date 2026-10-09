#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'config.yaml'));p.add_argument('--manifest',default=None);p.add_argument('--skip-contribution',action='store_true');a=p.parse_args()
 stages=['00_preflight.py','00_screen.py','01_bias_audit.py','02_blind_inversion.py','03_observability.py']
 if not a.skip_contribution:stages.append('04_contribution.py')
 for s in stages:
  cmd=[sys.executable,str(ROOT/'experiments'/s),'--config',a.config]
  if a.manifest:cmd+=['--manifest',a.manifest]
  print('==>',' '.join(cmd),flush=True);r=subprocess.run(cmd,cwd=ROOT)
  if r.returncode:return r.returncode
 subprocess.run([sys.executable,str(ROOT/'analysis'/'report.py'),'--config',a.config],cwd=ROOT)
 return 0
if __name__=='__main__':raise SystemExit(main())
