"""Check formal inputs and Hunyuan download access on a SLURM compute node."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request

from ai6132_benchmark.auth import hf_opener, read_hf_token
from ai6132_benchmark.dataset import load_canonical
from ai6132_benchmark.io import write_json


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Submit preflight_hunyuan.sh with sbatch')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--pilot-cases', type=Path)
    p.add_argument('--skip-pilot', action='store_true', help='Explicit user-authorized direct formal generation')
    p.add_argument('--token-file', type=Path, help='Private credential file; contents are never logged')
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'slurm_job_id': os.environ['SLURM_JOB_ID'], 'blockers': [], 'repositories': {}}
    rows, inputs = load_canonical(args.cases, decode=True)
    report['formal_inputs'] = {'cases': len(rows), **inputs}
    if args.pilot_cases:
        pilot, pilot_inputs = load_canonical(args.pilot_cases, decode=True, formal=False)
        if len(pilot) != 3 or {int(r['target_frames']) for r in pilot} != {129, 513}:
            raise ValueError('Independent pilot requires three short/long cases')
        if {v['sha256'] for v in pilot_inputs['images'].values()} & {
                v['sha256'] for v in inputs['images'].values()}:
            raise ValueError('Pilot images overlap formal inputs')
        report['pilot_inputs'] = {'cases': 3, **pilot_inputs}
    elif args.skip_pilot:
        report['pilot_validation'] = 'skipped_by_explicit_user_request'
    else:
        report['blockers'].append('Independent three-case short/long pilot manifest is not configured')
    runtime = args.runtime_root.resolve()
    report['runtime_root'] = str(runtime)
    prerequisites = ['env/bin/python', 'HunyuanVideo-1.5/.git', 'assets/assets.json']
    if not args.skip_pilot:
        prerequisites.append('outputs/pilot/pilot_pass.json')
    for relative in prerequisites:
        if not (runtime / relative).exists():
            report['blockers'].append(f'Missing runtime prerequisite: {relative}')
    token = read_hf_token(args.token_file)
    opener = hf_opener()
    headers = {'User-Agent': 'ai6132-hunyuan-preflight'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    report['hf_credential_available'] = bool(token)
    for repo in ('tencent/HunyuanVideo-1.5', 'Qwen/Qwen2.5-VL-7B-Instruct',
                 'google/byt5-small', 'black-forest-labs/FLUX.1-Redux-dev'):
        try:
            url = f'https://huggingface.co/api/models/{repo}?blobs=true'
            with opener.open(Request(url, headers=headers), timeout=45) as response:
                metadata = json.load(response)
            revision = metadata['sha']
            files = metadata['siblings']
            if repo == 'tencent/HunyuanVideo-1.5':
                chosen = [f for f in files if f['rfilename'].startswith(
                    ('transformer/480p_i2v_step_distilled/', 'vae/', 'scheduler/'))]
            elif repo == 'black-forest-labs/FLUX.1-Redux-dev':
                chosen = [f for f in files if f['rfilename'].startswith(
                    ('image_encoder/', 'feature_extractor/'))]
            else:
                chosen = [f for f in files if f['rfilename'].endswith(
                    ('.json', '.safetensors', '.bin', '.model', '.txt'))]
            item = {'revision': revision, 'gated': metadata.get('gated'),
                    'selected_bytes': sum(f.get('size', 0) for f in chosen),
                    'files': [{k: f[k] for k in ('rfilename', 'size') if k in f} for f in chosen]}
            report['repositories'][repo] = item
            weights = [f['rfilename'] for f in chosen if f['rfilename'].endswith(('.safetensors', '.bin', '.pt'))]
            if not weights:
                report['blockers'].append(f'No matching weight files identified: {repo}')
                continue
            probe = f'https://huggingface.co/{repo}/resolve/{revision}/{weights[0]}'
            try:
                with opener.open(Request(probe, headers=headers, method='HEAD'), timeout=45) as response:
                    item['weight_access_http'] = response.status
            except HTTPError as exc:
                item['weight_access_http'] = exc.code
                report['blockers'].append(f'Weight access denied or unavailable: {repo} (HTTP {exc.code})')
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            # Only error types/statuses are retained; URLs and credential headers are omitted.
            report['repositories'][repo] = {'error_type': type(exc).__name__,
                                            'http_status': getattr(exc, 'code', None)}
            report['blockers'].append(f'Model metadata unavailable: {repo} ({type(exc).__name__})')
    report['selected_bytes_subtotal'] = sum(
        r.get('selected_bytes', 0) for r in report['repositories'].values())
    report['storage_note'] = 'Subtotal excludes Glyph-SDXL-v2, the Conda environment, caches and video outputs'
    report['ready_for_formal'] = not report['blockers']
    write_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('formal_inputs', 'repositories')}, indent=2))
    print(json.dumps(report['repositories'], indent=2))
    return 0 if report['ready_for_formal'] else 2


if __name__ == '__main__':
    sys.exit(main())
