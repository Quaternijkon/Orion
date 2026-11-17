#!/usr/bin/env python3
"""Add three scoped benchmark routes; validate TCP routing; support exact rollback."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess

HOSTS = ('10.1.2.3', '10.1.2.4', '10.1.2.5')
GATEWAY = '10.1.1.254'
DEVICE = 'ens6f0np0'
STATE = Path('/run/orion-navigation-size-sift1m-direct-routes.json')
IP = '/usr/sbin/ip'


def ip(*args):
    return subprocess.check_output([IP, *args], text=True)


def routes(host):
    return [r for r in json.loads(ip('-j', 'route', 'show', 'table', 'main'))
            if r.get('dst') in (host, host + '/32')]


def exact(route):
    return route.get('gateway') == GATEWAY and route.get('dev') == DEVICE


def inspect(uid):
    results = {}
    for host in HOSTS:
        rows = json.loads(ip('-j', 'route', 'get', host, 'ipproto', 'tcp',
                             'dport', '29635', 'uid', str(uid)))
        results[host] = rows
    return results


def rollback(record):
    errors = []
    for host in reversed(record['added']):
        current = routes(host)
        if not current:
            continue
        if len(current) != 1 or not exact(current[0]):
            errors.append(f'{host}: route changed; refused removal')
            continue
        try:
            ip('route', 'del', host + '/32', 'via', GATEWAY, 'dev', DEVICE)
        except subprocess.CalledProcessError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError('; '.join(errors))
    STATE.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('apply', 'status', 'undo'))
    args = parser.parse_args()
    uid = int(os.environ.get('SUDO_UID', os.getuid()))
    if args.action == 'status':
        print(json.dumps(inspect(uid), indent=2))
        return
    if os.geteuid() != 0:
        raise SystemExit('Route changes require sudo authentication in your terminal.')
    if args.action == 'undo':
        if not STATE.exists():
            raise SystemExit('No owned route transaction exists; nothing removed.')
        rollback(json.loads(STATE.read_text()))
        print('Removed only routes added by this script.')
        return
    if STATE.exists():
        raise SystemExit('Transaction already exists. Use status or undo before applying again.')
    addresses = json.loads(ip('-j', '-4', 'address', 'show', 'dev', DEVICE))
    assert any(a.get('local') == '10.1.1.1' for x in addresses for a in x['addr_info']), 'Wrong machine'
    for host in HOSTS:
        current = routes(host)
        if current and (len(current) != 1 or not exact(current[0])):
            raise SystemExit(f'{host}: conflicting host route; no changes made')
    record = {'added': [], 'before': inspect(uid), 'uid': uid}
    with STATE.open('x') as stream:
        json.dump(record, stream)
    os.chmod(STATE, 0o600)
    try:
        for host in HOSTS:
            if not routes(host):
                # Record intent first so interrupted execution can be rolled back.
                record['added'].append(host)
                STATE.write_text(json.dumps(record, indent=2))
                ip('route', 'add', host + '/32', 'via', GATEWAY, 'dev', DEVICE)
        after = inspect(uid)
        for host, rows in after.items():
            assert rows and exact(rows[0]) and rows[0].get('prefsrc') == '10.1.1.1', (host, rows)
            with socket.create_connection((host, 22), timeout=5) as connection:
                assert connection.getsockname()[0] == '10.1.1.1'
        record['after'] = after
        STATE.write_text(json.dumps(record, indent=2))
        print(json.dumps({'verified': True, 'routes': after}, indent=2))
    except BaseException:
        rollback(record)
        raise


if __name__ == '__main__':
    main()
