"""Launch the two YAML-defined experiments sequentially in a detached supervisor."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ('predictive_ghost','lifetime_ghost')


def write_json(path,payload):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    os.chdir(ROOT)
    paths=[ROOT/'checkpoints'/name for name in EXPERIMENTS]
    for path in paths:
        path.mkdir(parents=True,exist_ok=True)
    # One lock covers both experiment folders and remains held by the supervisor.
    lock=(paths[0]/'study.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A patch/ghost study is already running')
    if not args.worker:
        environment=dict(os.environ)
        environment.pop('LD_LIBRARY_PATH',None);environment.pop('LD_PRELOAD',None)
        environment['PYTHONUNBUFFERED']='1'
        # Worker takes over the lock; its PID/logs are recorded before it trains.
        fcntl.flock(lock,fcntl.LOCK_UN)
        with (paths[0]/'study.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-m','scripts.run_patch_ghost_study','--worker'],cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for path in paths:
            (path/'study.pid').write_text(str(process.pid)+'\n')
        print(f'Launched sequential study supervisor PID {process.pid}; logs: {paths[0]/"study.log"}')
        return
    import yaml
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    plan=[]
    for name,path in zip(EXPERIMENTS,paths):
        trainer=importlib.import_module(f'src.{name}.training')
        config=trainer.load_config(ROOT/'src'/name/'config.yaml')
        for family in config['study']['families']:
            for policy in config['study']['policies']:
                for seed in config['study']['seeds']:
                    c=deepcopy(config); c['model']['family']=family;c['model']['routing']['policy']=policy;c['seed']=seed
                    snapshot=path/'study_configs'/stamp/f'{family}_{policy}_seed{seed}.yaml'
                    snapshot.parent.mkdir(parents=True,exist_ok=True)
                    snapshot.write_text(yaml.safe_dump(c,sort_keys=False))
                    trainer.load_config(snapshot)
                    plan.append(dict(experiment=name,family=family,policy=policy,seed=seed,config=str(snapshot.relative_to(ROOT)),status='queued'))
    state=dict(timestamp=stamp,pid=os.getpid(),status='running',runs=plan)
    def save():
        for path in paths:
            write_json(path/'study.json',state)
    save()
    for item in plan:
        item['status']='running';save()
        print('Starting',item,flush=True)
        path=ROOT/'checkpoints'/item['experiment']
        before=set((path/'runs').glob('*')) if (path/'runs').exists() else set()
        with (path/'study.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-m',f'src.{item["experiment"]}.training','--config',item['config']],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            item['pid']=process.pid;save()
            code=process.wait()
        new=set((path/'runs').glob('*'))-before
        item['outputs']=[str(p.relative_to(ROOT)) for p in sorted(new)]
        item['exit_code']=code
        if code:
            item['status']='failed';state['status']='failed';save()
            raise SystemExit(f'Study stopped after failed run: {item}')
        item['status']='complete'
        for directory in new:
            if (directory/'result.json').is_file():
                item['result']=json.loads((directory/'result.json').read_text())
        save()
    state['status']='complete';save()
    for path in paths:
        write_json(path/'comparison.json',state)
    print('Sequential study complete',flush=True)


if __name__=='__main__':
    main()
