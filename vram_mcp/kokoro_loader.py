"""Optional Windows-only ownership of one reviewed Hermes Kokoro CUDA worker.

This is a launch/settlement capability, not a synthesis service or an arbitrary
command runner. Configuration is operator-owned; tool callers supply no paths.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import threading
from pathlib import Path

from . import claims, core

# Reviewed Hermes worker: parent HANDLE watchdog starts before any CUDA import.
# Source authority: sushiHex/hermes-realtime, worker at 70c79e1dc88cc6dfc9ea356f5b6f1141037bfd3e.
WORKER_SHA256 = 'ec6324e32b543f32a60ac3d7eed3c4ac686e564249454d768d68cc21bd5db20f'
MODEL_SHA256 = 'beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a'
VOICES_SHA256 = 'bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d'
_TOKEN_ENV = 'HERMES_KOKORO_WORKER_TOKEN'
WORKER_EXIT_CATEGORIES = {
    80: 'authentication_failed', 81: 'parent_watchdog_failed',
    82: 'asset_attestation_failed', 83: 'dependency_import_failed',
    84: 'dll_preload_failed', 85: 'cuda_session_failed',
    86: 'cuda_provider_rejected', 87: 'kokoro_engine_failed',
    88: 'attestation_synthesis_failed', 89: 'cuda_profile_unavailable_or_invalid',
    90: 'cuda_profile_missing_core_ops', 91: 'cuda_profile_forbidden_cpu_ops',
    92: 'loopback_connection_failed', 93: 'worker_service_failed',
}
_TTL = 120
_LABEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z')
_TOKEN = re.compile(r'[A-Za-z0-9_-]{32,128}\Z')
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_FIELDS = {'version', 'python', 'python_sha256', 'venv_python', 'venv_python_sha256',
           'worker', 'worker_sha256', 'model', 'model_sha256', 'voices', 'voices_sha256',
           'gpu_index', 'gpu_uuid', 'reserve_gb', 'headroom_mb', 'voice', 'language'}
_ENV = ('SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'PROGRAMFILES',
        'PROGRAMFILES(X86)', 'USERPROFILE', 'HOME', 'LOCALAPPDATA', 'APPDATA')


class Refusal(ValueError):
    """Fixed, payload-free failure category."""


def require(condition, category):
    if not condition:
        raise Refusal(category)


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'profile_schema')
        result[key] = value
    return result


def read_profile(path):
    require(path is not None, 'not_configured')
    with Path(path).open('rb') as stream:
        raw = stream.read(16385)
    require(len(raw) <= 16384, 'profile_bound')
    p = json.loads(raw, object_pairs_hook=_unique)
    require(type(p) is dict and set(p) == _FIELDS and type(p['version']) is int
            and p['version'] == 1, 'profile_schema')
    for name in ('python', 'venv_python', 'worker', 'model', 'voices'):
        value, digest = p[name], p[name + '_sha256']
        require(type(value) is str and type(digest) is str and _HEX.fullmatch(digest),
                'profile_schema')
        item = Path(value)
        require(item.is_absolute() and item.is_file() and not item.is_symlink(),
                'profile_file')
        require(_digest(item) == digest, 'profile_identity')
        p[name] = item.resolve(strict=True)
    source = p['worker'].read_bytes().replace(b'\r\n', b'\n')
    require(hashlib.sha256(source).hexdigest() == WORKER_SHA256, 'unsupported_worker')
    require(p['model_sha256'] == MODEL_SHA256 and p['voices_sha256'] == VOICES_SHA256,
            'unsupported_assets')
    require(type(p['gpu_index']) is int and p['gpu_index'] >= 0
            and type(p['gpu_uuid']) is str
            and re.fullmatch(r'GPU-[A-Za-z0-9-]{8,80}', p['gpu_uuid']), 'profile_device')
    gb = p['reserve_gb']
    require(type(gb) in (int, float) and math.isfinite(gb) and 0 < gb <= 128,
            'profile_capacity')
    require(type(p['headroom_mb']) is int and 1024 <= p['headroom_mb'] <= 131072,
            'profile_capacity')
    require(p['voice'] == 'bf_isabella' and p['language'] == 'en-gb', 'profile_voice')
    return p


def verify_interpreter(profile, environment):
    """Reject venv redirector custody: the Popen image must be base Python itself."""
    code = (
        'import ctypes,json,sys;'
        'b=ctypes.create_unicode_buffer(32768);'
        'n=ctypes.windll.kernel32.GetModuleFileNameW(None,b,len(b));'
        'print(json.dumps([b.value if n else None,sys._base_executable,sys.executable]))'
    )
    result = subprocess.run([str(profile['python']), '-I', '-B', '-S', '-c', code],
                            stdin=subprocess.DEVNULL, capture_output=True, timeout=10,
                            env=environment, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(result.returncode == 0 and len(result.stdout) <= 4096, 'interpreter_binding')
    values = json.loads(result.stdout)
    require(type(values) is list and len(values) == 3 and all(type(x) is str for x in values),
            'interpreter_binding')
    require(Path(values[0]).resolve() == profile['python']
            and Path(values[1]).resolve() == profile['python']
            and Path(values[2]).resolve() == profile['venv_python'], 'interpreter_binding')


def _result(outcome, state, reason=None, handle=None, record=None):
    return {'ok': outcome == 'succeeded', 'outcome': outcome, 'state': state,
            'reason': reason, 'operation_id': handle, 'gpu_ready': False,
            'worker_exit_code': record.get('worker_exit_code') if record else None,
            'worker_exit_category': record.get('worker_exit_category') if record else None}


class KokoroLoader:
    def __init__(self, profile_path, *, gpu_reading, spawn=subprocess.Popen,
                 ledger_path=None, platform=sys.platform, monitor=True,
                 interpreter_probe=verify_interpreter):
        self.profile_path = profile_path
        self.gpu_reading, self.spawn = gpu_reading, spawn
        self.ledger_path, self.platform = ledger_path, platform
        self.interpreter_probe = interpreter_probe
        self._lock = threading.RLock()
        self._records = {}
        self._closed = False
        self._shutdown_required = False
        self._wake = threading.Event()
        self._thread = None
        self._monitor = monitor

    def _ledger(self, method, *args, **kwargs):
        return getattr(claims, method)(*args, path=self.ledger_path, **kwargs)

    def load(self, owner, request_id, connect_port, token):
        """Return launch acceptance only; Hermes independently checks CUDA readiness."""
        with self._lock:
            reservation = operation = None
            record = None
            answer = None
            try:
                require(not self._closed, 'closed')
                require(not self._shutdown_required, 'scoped_server_shutdown_required')
                require(self.platform == 'win32', 'unsupported_platform')
                require(type(owner) is str and _LABEL.fullmatch(owner), 'request_schema')
                require(type(request_id) is str and _LABEL.fullmatch(request_id), 'request_schema')
                require(type(connect_port) is int and 1024 <= connect_port <= 65535,
                        'request_schema')
                require(type(token) is str and _TOKEN.fullmatch(token), 'request_schema')
                binding = hashlib.sha256(f'{connect_port}:{token}'.encode()).hexdigest()
                self._refresh_locked()
                for handle, existing in self._records.items():
                    if existing['owner'] == owner and existing['request'] == request_id:
                        require(existing['binding'] == binding, 'request_mismatch')
                        return self._view(handle, existing)
                require(not any(r['state'] != 'stopped' for r in self._records.values()),
                        'worker_active')
                require(len(self._records) < 64, 'request_history_full')
                p = read_profile(self.profile_path)
                environment = {k: v for k, v in os.environ.items() if k.upper() in _ENV}
                environment['__PYVENV_LAUNCHER__'] = str(p['venv_python'])
                environment['CUDA_VISIBLE_DEVICES'] = p['gpu_uuid']
                environment['PYTHONDONTWRITEBYTECODE'] = '1'
                self.interpreter_probe(p, environment)
                try:
                    reservation = self._ledger('reserve', p['reserve_gb'], 'kokoro-loader',
                                               'owned Kokoro worker', ttl_seconds=_TTL)['claim_id']
                except (ValueError, OSError):
                    raise Refusal('admission_unavailable') from None
                operation = self._ledger('begin_operation', 'vram-mcp/kokoro', 'warm',
                                         scope=f"gpu:index={p['gpu_index']}")
                require(operation.get('ok') is True, 'admission_pending')
                reading = self.gpu_reading()
                require(reading.known and type(reading.data) is list and len(reading.data) == 1,
                        'gpu_unknown')
                gpu = reading.data[0]
                require(type(gpu) is dict and type(gpu.get('index')) is int
                        and gpu['index'] == p['gpu_index'] and gpu.get('uuid') == p['gpu_uuid'],
                        'device_mismatch')
                free = gpu.get('free_mb')
                require(type(free) in (int, float) and math.isfinite(free) and free >= 0,
                        'gpu_unknown')
                active = self._ledger('list_claims')
                require(any(r.get('claim_id') == reservation for r in active),
                        'reservation_missing')
                require(free - core.reserved_mb(active) >= p['headroom_mb'],
                        'insufficient_capacity')
                handle = secrets.token_urlsafe(32)
                record = {'owner': owner, 'request': request_id, 'binding': binding,
                          'reservation': reservation, 'process': None, 'state': 'launching',
                          'failure': None, 'spawn_unknown': False,
                          'worker_exit_code': None, 'worker_exit_category': None,
                          'termination_started': False,
                          'admission': operation['operation_id']}
                self._records[handle] = record
                if self._monitor and self._thread is None:
                    thread = threading.Thread(target=self._watch, daemon=True,
                                              name='owned-kokoro')
                    thread.start()
                    self._thread = thread
                environment[_TOKEN_ENV] = token
                command = [str(p['python']), '-I', '-B', str(p['worker']),
                           '--connect-port', str(connect_port), '--parent-pid', str(os.getpid()),
                           '--model', str(p['model']), '--model-sha256', p['model_sha256'],
                           '--voices', str(p['voices']), '--voices-sha256', p['voices_sha256'],
                           '--attestation-voice', p['voice'], '--language', p['language'],
                           '--max-pcm-bytes', str(16 * 1024 * 1024)]
                try:
                    record['process'] = self.spawn(command, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment,
                        shell=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                except BaseException:
                    # A spawn exception is not a kernel witness that no child
                    # was created. Keep capacity and forbid a duplicate launch.
                    record['spawn_unknown'] = True
                    record['state'] = 'cleanup_unknown'
                    raise
                if not self._ledger('renew', reservation, ttl_seconds=_TTL).get('ok'):
                    raise RuntimeError('reservation_lost')
                record['state'] = 'launch_accepted'
                answer = self._view(handle, record)
            except Refusal as exc:
                answer = _result('refused', 'refused', str(exc))
            except Exception:
                if record is not None and record['spawn_unknown']:
                    answer = self._view(handle, record)
                elif record is not None and record['process'] is not None:
                    record['failure'] = 'launch_failed'
                    self._settle(record)
                    answer = self._view(handle, record)
                else:
                    if record is not None:
                        record['state'] = 'stopped'
                        record['failure'] = 'launch_failed'
                        answer = self._view(handle, record)
                    else:
                        answer = _result('failed', 'refused', 'launch_failed')
            finally:
                if operation is not None and operation.get('ok') is True:
                    if record is not None and record['state'] == 'cleanup_unknown':
                        # Keep the existing OS-backed warm lease when startup
                        # cannot prove settlement, even if its claim expired.
                        record['admission'] = operation['operation_id']
                    else:
                        try:
                            if record is None:
                                self._finish_admission({'admission': operation['operation_id']})
                            else:
                                self._finish_admission(record)
                        except Exception:
                            if record is not None:
                                self._settle(record)
                                record['failure'] = 'admission_finalize_failed'
                                answer = self._view(handle, record)
                            else:
                                self._shutdown_required = True
                                answer = _result('unknown', 'cleanup_unknown',
                                                 'scoped_server_shutdown_required')
                if reservation is not None and (record is None or (
                        record['process'] is None and not record['spawn_unknown'])):
                    try:
                        self._ledger('release', reservation)
                    except Exception:
                        if not self._shutdown_required:
                            answer = _result('unknown', 'cleanup_unknown', 'reservation_release_unproven')
            if record is not None and record['state'] == 'cleanup_unknown':
                return self._view(handle, record)
            return answer

    def _view(self, handle, record):
        state = record['state']
        if state == 'stopped' and record.get('failure'):
            return _result('failed', state, record['failure'], handle, record)
        if record.get('spawn_unknown'):
            return _result('unknown', 'cleanup_unknown', 'launch_outcome_unknown', handle)
        return _result('unknown' if state == 'cleanup_unknown' else 'succeeded',
                       state, 'cleanup_unproven' if state == 'cleanup_unknown' else None,
                       handle, record)

    def status(self, owner, handle):
        with self._lock:
            record = self._records.get(handle) if type(handle) is str else None
            if record is None or record['owner'] != owner:
                return _result('refused', 'refused', 'operation_not_owned')
            self._refresh_locked((record,))
            return self._view(handle, record)

    def stop(self, owner, handle):
        with self._lock:
            record = self._records.get(handle) if type(handle) is str else None
            if record is None or record['owner'] != owner:
                return _result('refused', 'refused', 'operation_not_owned')
            self._settle(record)
            return self._view(handle, record)

    def cancel_request(self, owner, request_id):
        """A canceled MCP request must not lose custody of an accepted launch."""
        with self._lock:
            for record in self._records.values():
                if record['owner'] == owner and record['request'] == request_id:
                    self._settle(record)

    def _settle(self, record):
        if record.get('spawn_unknown'):
            record['state'] = 'cleanup_unknown'
            return
        process = record['process']
        try:
            code = process.poll() if process is not None else None
            self._capture_exit(record, code)
            if process is not None and code is None:
                # A raced exit after this point cannot establish a natural
                # worker category, even if terminate itself raises.
                record['termination_started'] = True
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            code = process.poll() if process is not None else None
            self._capture_exit(record, code)
            if process is not None and code is None:
                raise RuntimeError('still_running')
            self._ledger('release', record['reservation'])
            self._finish_admission(record)
            record['process'] = None
            record['state'] = 'stopped'
        except Exception:
            record['state'] = 'cleanup_unknown'

    @staticmethod
    def _capture_exit(record, code):
        if (type(code) is int and -(2 ** 31) <= code <= 2 ** 32 - 1
                and record['worker_exit_code'] is None):
            record['worker_exit_code'] = code
            if not record['termination_started']:
                record['worker_exit_category'] = WORKER_EXIT_CATEGORIES.get(code)

    def _finish_admission(self, record):
        operation = record.get('admission')
        if operation is not None:
            result = self._ledger('finish_operation', operation)
            if type(result.get('ok')) is not bool or result.get('retained') is not False:
                raise RuntimeError('admission_release_unproven')
            record['admission'] = None

    def _refresh_locked(self, records=None):
        for record in self._records.values() if records is None else records:
            if record['state'] == 'stopped':
                continue
            process = record['process']
            try:
                if record.get('spawn_unknown'):
                    self._ledger('renew', record['reservation'], ttl_seconds=_TTL)
                elif process is None or process.poll() is not None:
                    self._settle(record)
                elif not self._ledger('renew', record['reservation'], ttl_seconds=_TTL).get('ok'):
                    self._settle(record)
            except Exception:
                self._settle(record)

    def refresh(self):
        with self._lock:
            self._refresh_locked()

    def _watch(self):
        while not self._wake.wait(15):
            self.refresh()

    def close(self):
        self._wake.set()
        with self._lock:
            self._closed = True
            for record in self._records.values():
                if record['state'] != 'stopped':
                    self._settle(record)
        if self._thread is not None:
            self._thread.join(timeout=2)
