#!/usr/bin/env python3
from pathlib import Path
import argparse,ast,subprocess,sys,json,importlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
REQ=['oc_arcpith_rg/geometry.py','oc_arcpith_rg/objectives.py','oc_arcpith_rg/candidates.py','oc_arcpith_rg/screen.py','oc_arcpith_rg/bias.py','oc_arcpith_rg/cv.py','oc_arcpith_rg/optimizer.py','oc_arcpith_rg/observability.py','oc_arcpith_rg/contribution.py','experiments/00_screen.py','experiments/01_bias_audit.py','experiments/02_blind_inversion.py','experiments/03_observability.py','experiments/04_contribution.py','EXPERIMENT_GUIDE.md','AUDIT_REPORT.md']

def internal_import_audit():
    missing=[]
    pkg=ROOT/'oc_arcpith_rg'
    for p in pkg.glob('*.py'):
        tree=ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom) and n.level>0 and n.module:
                target=pkg/(n.module.split('.')[0]+'.py')
                if not target.exists():missing.append({'file':str(p.relative_to(ROOT)),'module':n.module})
    return missing

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    missing=[x for x in REQ if not (ROOT/x).exists()];bad=[]
    py=list((ROOT/'oc_arcpith_rg').glob('*.py'))+list((ROOT/'experiments').glob('*.py'))+list((ROOT/'analysis').glob('*.py'))+list((ROOT/'tools').glob('*.py'))
    for p in py:
        try:ast.parse(p.read_text())
        except Exception as e:bad.append([str(p.relative_to(ROOT)),repr(e)])
    comp=subprocess.run([sys.executable,'-m','compileall','-q','oc_arcpith_rg','experiments','analysis','tools','tests'],cwd=ROOT).returncode
    import_errors=[];imported=[]
    for p in (ROOT/'oc_arcpith_rg').glob('*.py'):
        if p.name=='__init__.py':continue
        name='oc_arcpith_rg.'+p.stem
        try:importlib.import_module(name);imported.append(name)
        except Exception as e:import_errors.append([name,repr(e)])
    tst=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    smoke=None
    if a.smoke:
        out=ROOT/'results_validation';
        if out.exists():
            import shutil;shutil.rmtree(out)
        # temporary config pointing to isolated result root
        import yaml
        cfg=yaml.safe_load((ROOT/'config.yaml').read_text());cfg['paths']['results_dir']=str(out)
        tmp=ROOT/'config.validation.yaml';tmp.write_text(yaml.safe_dump(cfg,sort_keys=False))
        r=subprocess.run([sys.executable,str(ROOT/'experiments'/'run_all.py'),'--config',str(tmp),'--manifest',str(ROOT/'data'/'demo'/'manifest.jsonl')],cwd=ROOT,capture_output=True,text=True)
        smoke={'returncode':r.returncode,'stdout_tail':r.stdout[-3000:],'stderr_tail':r.stderr[-3000:],'final_report_exists':(out/'FINAL_REPORT.md').exists(),'contribution_exists':(out/'contribution'/'ring_contribution.jsonl').exists()}
        tmp.unlink(missing_ok=True)
    out={'missing_required_files':missing,'missing_internal_imports':internal_import_audit(),'syntax_errors':bad,'compileall':comp==0,'imported_modules':len(imported),'import_errors':import_errors,'tests_returncode':tst.returncode,'tests_output':(tst.stdout+tst.stderr)[-4000:],'smoke':smoke}
    print(json.dumps(out,indent=2));fail=missing or out['missing_internal_imports'] or bad or comp or import_errors or tst.returncode or (smoke is not None and (smoke['returncode']!=0 or not smoke['final_report_exists'] or not smoke['contribution_exists']));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
