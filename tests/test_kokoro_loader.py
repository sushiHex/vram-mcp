"""Owned Kokoro launch boundaries; no GPU, network, or model execution."""
import hashlib
import json
import subprocess

import pytest

from vram_mcp import claims, kokoro_loader
from vram_mcp.observations import Observation


class Process:
    def __init__(self, *, stubborn=False):
        self.code = None
        self.stubborn = stubborn
        self.terminated = self.killed = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.code = -15

    def kill(self):
        self.killed = True
        if not self.stubborn:
            self.code = -9

    def wait(self, timeout):
        if self.code is None:
            raise subprocess.TimeoutExpired('owned-worker', timeout)
        return self.code


@pytest.fixture
def rig(tmp_path, monkeypatch):
    files = {}
    for name in ('python.exe', 'venv-python.exe', 'kokoro_worker.py', 'model.onnx', 'voices.bin'):
        path = tmp_path / name
        path.write_bytes(('fixture-' + name).encode())
        files[name] = path
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(kokoro_loader, 'WORKER_SHA256', digest(files['kokoro_worker.py']))
    monkeypatch.setattr(kokoro_loader, 'MODEL_SHA256', digest(files['model.onnx']))
    monkeypatch.setattr(kokoro_loader, 'VOICES_SHA256', digest(files['voices.bin']))
    profile = {
        'version': 1, 'python': str(files['python.exe']),
        'python_sha256': digest(files['python.exe']),
        'venv_python': str(files['venv-python.exe']),
        'venv_python_sha256': digest(files['venv-python.exe']),
        'worker': str(files['kokoro_worker.py']),
        'worker_sha256': digest(files['kokoro_worker.py']),
        'model': str(files['model.onnx']), 'model_sha256': digest(files['model.onnx']),
        'voices': str(files['voices.bin']), 'voices_sha256': digest(files['voices.bin']),
        'gpu_index': 0, 'gpu_uuid': 'GPU-fixture-0001', 'reserve_gb': 1.0,
        'headroom_mb': 1024, 'voice': 'bf_isabella', 'language': 'en-gb',
    }
    profile_path = tmp_path / 'operator.json'
    profile_path.write_text(json.dumps(profile))
    process = Process()
    calls = []
    reading = Observation([{'index': 0, 'uuid': 'GPU-fixture-0001', 'free_mb': 4096}],
                          'fake', scope='gpu:index=0')
    def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        return process
    manager = kokoro_loader.KokoroLoader(
        profile_path, gpu_reading=lambda: reading, spawn=spawn,
        ledger_path=tmp_path / 'claims.json', platform='win32', monitor=False,
        interpreter_probe=lambda *args: None,
    )
    original_finish = claims.finish_operation
    yield manager, profile, profile_path, process, calls, reading
    manager.close()
    # A refused/unknown native process has no real child in these mocks. Drop
    # retained private admission locks as kernel server exit would do, without
    # claiming successful worker cleanup or clearing its unknown receipt.
    for record in manager._records.values():
        if record.get('admission'):
            original_finish(record['admission'], uncertain=True, path=manager.ledger_path)


def load(manager, request='request-one', token='x' * 43):
    return manager.load('hermes-preview', request, 12345, token)


def test_load_owns_cpu_ineligible_worker_and_device_without_claiming_gpu_ready(rig):
    manager, _, _, process, calls, _ = rig
    result = load(manager)
    assert result['outcome'] == 'succeeded'
    assert result['state'] == 'launch_accepted' and result['gpu_ready'] is False
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0][1] == '-I'
    assert kwargs['shell'] is False
    assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-fixture-0001'
    assert kwargs['env']['HERMES_KOKORO_WORKER_TOKEN'] == 'x' * 43
    assert kwargs['stdout'] == kwargs['stderr'] == subprocess.DEVNULL
    assert '--parent-pid' in args[0]
    assert claims.list_claims(path=manager.ledger_path)
    stopped = manager.stop('hermes-preview', result['operation_id'])
    assert stopped['state'] == 'stopped' and process.poll() is not None
    assert claims.list_claims(path=manager.ledger_path) == []


@pytest.mark.parametrize('change', ('unknown_gpu', 'wrong_device', 'capacity', 'infinite_capacity', 'changed_asset',
                                  'unsupported_worker', 'unsupported_assets', 'platform',
                                  'extra_profile'))
def test_refused_load_never_spawns_and_leaves_no_owned_reservation(rig, change):
    manager, profile, profile_path, _, calls, reading = rig
    if change == 'unknown_gpu':
        manager.gpu_reading = lambda: Observation(reading.data, 'fake', error='unavailable')
    elif change == 'wrong_device':
        manager.gpu_reading = lambda: Observation([{'index': 1, 'uuid': 'GPU-other',
                                                   'free_mb': 4096}], 'fake')
    elif change == 'capacity':
        manager.gpu_reading = lambda: Observation([{'index': 0, 'uuid': profile['gpu_uuid'],
                                                   'free_mb': 1024}], 'fake')
    elif change == 'infinite_capacity':
        manager.gpu_reading = lambda: Observation([{'index': 0, 'uuid': profile['gpu_uuid'],
                                                   'free_mb': float('inf')}], 'fake')
    elif change == 'changed_asset':
        from pathlib import Path
        Path(profile['model']).write_bytes(b'changed')
    elif change == 'unsupported_worker':
        monkey_digest = hashlib.sha256(b'unreviewed-worker').hexdigest()
        from pathlib import Path
        Path(profile['worker']).write_bytes(b'unreviewed-worker')
        profile['worker_sha256'] = monkey_digest
        profile_path.write_text(json.dumps(profile))
    elif change == 'unsupported_assets':
        from pathlib import Path
        Path(profile['model']).write_bytes(b'unsupported model')
        profile['model_sha256'] = hashlib.sha256(b'unsupported model').hexdigest()
        profile_path.write_text(json.dumps(profile))
    elif change == 'platform':
        manager.platform = 'linux'
    else:
        profile['command'] = 'not allowed'
        profile_path.write_text(json.dumps(profile))
    assert load(manager)['outcome'] == 'refused'
    assert calls == []
    assert claims.list_claims(path=manager.ledger_path) == []


def test_other_reservations_and_pending_admission_are_preserved(rig):
    manager, _, _, _, calls, _ = rig
    other = claims.reserve(3, 'personal', 'existing', path=manager.ledger_path)
    assert load(manager)['outcome'] == 'refused'
    assert [r['claim_id'] for r in claims.list_claims(path=manager.ledger_path)] == [other['claim_id']]
    assert calls == []
    claims.release(other['claim_id'], path=manager.ledger_path)
    pending = claims.begin_operation('personal', 'warm', path=manager.ledger_path)
    try:
        assert load(manager)['outcome'] == 'refused'
        assert calls == []
    finally:
        claims.finish_operation(pending['operation_id'], path=manager.ledger_path)


def test_repeated_request_does_not_spawn_again_and_other_owners_cannot_stop(rig):
    manager, _, _, process, calls, _ = rig
    result = load(manager)
    assert load(manager) == result
    assert load(manager, token='y' * 43)['outcome'] == 'refused'
    assert manager.stop('other-owner', result['operation_id'])['outcome'] == 'refused'
    assert process.poll() is None and len(calls) == 1
    assert manager.status('hermes-preview', result['operation_id'])['gpu_ready'] is False


def test_uncertain_cleanup_keeps_reservation_and_refuses_another_load(rig):
    manager, _, _, process, calls, _ = rig
    process.stubborn = True
    result = load(manager)
    assert manager.stop('hermes-preview', result['operation_id'])['outcome'] == 'unknown'
    assert claims.list_claims(path=manager.ledger_path)
    assert load(manager, 'request-two')['outcome'] == 'refused'
    assert len(calls) == 1
    process.code = -9
    assert manager.status('hermes-preview', result['operation_id'])['state'] == 'stopped'
    assert claims.list_claims(path=manager.ledger_path) == []


def test_capacity_released_only_after_owned_child_exits(rig, monkeypatch):
    manager, _, _, process, _, _ = rig
    result = load(manager)
    release = claims.release
    def checked_release(*args, **kwargs):
        assert process.poll() is not None
        return release(*args, **kwargs)
    monkeypatch.setattr(claims, 'release', checked_release)
    assert manager.stop('hermes-preview', result['operation_id'])['state'] == 'stopped'


@pytest.mark.parametrize('where', ('spawn', 'probe'))
def test_private_dependency_errors_and_tokens_are_never_returned(rig, where):
    manager, _, _, _, _, _ = rig
    def fail(*args, **kwargs):
        raise OSError('private token=' + 'x' * 43)
    if where == 'spawn':
        manager.spawn = fail
    else:
        manager.interpreter_probe = fail
    result = load(manager)
    assert result['outcome'] == ('unknown' if where == 'spawn' else 'failed')
    assert 'private' not in json.dumps(result) and 'x' * 43 not in json.dumps(result)
    assert bool(claims.list_claims(path=manager.ledger_path)) == (where == 'spawn')


def test_child_exit_and_failed_renewal_settle_before_releasing(rig, monkeypatch):
    manager, _, _, process, _, _ = rig
    result = load(manager)
    monkeypatch.setattr(claims, 'renew', lambda *args, **kwargs: {'ok': False})
    manager.refresh()
    assert process.poll() is not None
    assert manager.status('hermes-preview', result['operation_id'])['state'] == 'stopped'
    assert claims.list_claims(path=manager.ledger_path) == []


def test_interpreter_probe_rejects_redirector_or_unknown_image_before_launch(rig):
    manager, _, _, _, calls, _ = rig
    def refuse(*args):
        raise kokoro_loader.Refusal('interpreter_binding')
    manager.interpreter_probe = refuse
    assert load(manager)['reason'] == 'interpreter_binding'
    assert calls == [] and claims.list_claims(path=manager.ledger_path) == []


def test_interpreter_probe_uses_isolated_site_free_process_and_actual_image(rig, monkeypatch):
    _, p, _, _, _, _ = rig
    from pathlib import Path
    p = {**p, 'python': Path(p['python']), 'venv_python': Path(p['venv_python'])}
    calls = []
    def probe(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps([
            str(p['python']), str(p['python']), str(p['venv_python']),
        ]).encode(), b'')
    monkeypatch.setattr(kokoro_loader.subprocess, 'run', probe)
    kokoro_loader.verify_interpreter(p, {'__PYVENV_LAUNCHER__': str(p['venv_python'])})
    assert calls[0][0][:5] == [str(p['python']), '-I', '-B', '-S', '-c']
    def redirector(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, json.dumps([
            'different-image.exe', str(p['python']), str(p['venv_python']),
        ]).encode(), b'')
    monkeypatch.setattr(kokoro_loader.subprocess, 'run', redirector)
    with pytest.raises(kokoro_loader.Refusal, match='interpreter_binding'):
        kokoro_loader.verify_interpreter(p, {})


def test_supported_worker_accepts_crlf_only_with_matching_raw_operator_hash(rig):
    manager, p, profile_path, _, calls, _ = rig
    from pathlib import Path
    worker = Path(p['worker'])
    original = worker.read_bytes()
    worker.write_bytes(original + b'\r\n')
    # Pin the reviewed LF form, but keep an independent exact on-disk digest.
    old_supported = kokoro_loader.WORKER_SHA256
    kokoro_loader.WORKER_SHA256 = hashlib.sha256(original + b'\n').hexdigest()
    try:
        assert load(manager)['outcome'] == 'refused'  # stale raw binding
        assert calls == []
        p['worker_sha256'] = hashlib.sha256(worker.read_bytes()).hexdigest()
        profile_path.write_text(json.dumps(p))
        assert load(manager)['outcome'] == 'succeeded'
    finally:
        kokoro_loader.WORKER_SHA256 = old_supported


def test_request_cancellation_settles_matching_child_only(rig):
    manager, _, _, process, _, _ = rig
    result = load(manager)
    manager.cancel_request('other-owner', 'request-one')
    assert process.poll() is None
    manager.cancel_request('hermes-preview', 'request-one')
    assert process.poll() is not None
    assert manager.status('hermes-preview', result['operation_id'])['state'] == 'stopped'


def test_finalize_error_does_not_return_stale_launch_acceptance(rig, monkeypatch):
    manager, _, _, process, _, _ = rig
    finish = claims.finish_operation
    def fail(*args, **kwargs):
        finish(*args, **kwargs)
        raise OSError('private error')
    monkeypatch.setattr(claims, 'finish_operation', fail)
    assert load(manager)['outcome'] != 'succeeded'
    assert process.poll() is not None
    assert claims.list_claims(path=manager.ledger_path) == []


def test_owned_load_does_not_forward_unrelated_credentials(rig, monkeypatch):
    manager, _, _, _, calls, _ = rig
    monkeypatch.setenv('OPENAI_API_KEY', 'private credential')
    load(manager)
    assert 'OPENAI_API_KEY' not in calls[0][1]['env']


def test_server_tools_forward_private_token_without_returning_or_auditing_it(rig, monkeypatch):
    import asyncio
    from vram_mcp import server
    manager, _, _, _, _, _ = rig
    monkeypatch.setattr(server, '_kokoro_loader', manager)
    def no_audit(*args, **kwargs):
        raise AssertionError('Kokoro request must not enter action audit')
    monkeypatch.setattr(server._audit, 'log_action', no_audit)
    result = asyncio.run(server.load_kokoro('hermes-preview', 'request-one', 12345, 'x' * 43))
    assert result['state'] == 'launch_accepted'
    assert 'x' * 43 not in json.dumps(result)
    stopped = asyncio.run(server.stop_kokoro('hermes-preview', result['operation_id']))
    assert stopped['state'] == 'stopped'
    refused = asyncio.run(server.load_kokoro('hermes-preview', 'bad', 12345,
                                           {'secret': 'never echo this'}))
    assert refused['reason'] == 'request_schema'
    assert 'never echo' not in json.dumps(refused)


def test_server_shutdown_closes_owned_worker_even_on_transport_error(rig, monkeypatch):
    from vram_mcp import server
    manager, _, _, process, _, _ = rig
    load(manager)
    monkeypatch.setattr(server, '_kokoro_loader', manager)
    monkeypatch.setattr(server, '_paging', None)
    def closed_transport():
        raise RuntimeError('transport closed')
    monkeypatch.setattr(server.mcp, 'run', closed_transport)
    with pytest.raises(RuntimeError, match='transport closed'):
        server.main()
    assert process.poll() is not None and claims.list_claims(path=manager.ledger_path) == []


def test_canceled_mcp_load_settles_worker_started_by_background_thread(rig, monkeypatch):
    import asyncio
    import threading
    from vram_mcp import server
    manager, _, _, process, _, _ = rig
    started, proceed = threading.Event(), threading.Event()
    def delayed_spawn(*args, **kwargs):
        started.set()
        assert proceed.wait(3)
        return process
    manager.spawn = delayed_spawn
    monkeypatch.setattr(server, '_kokoro_loader', manager)
    async def scenario():
        task = asyncio.create_task(server.load_kokoro('hermes-preview', 'request-one',
                                                      12345, 'x' * 43))
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert process.poll() is not None and claims.list_claims(path=manager.ledger_path) == []


def test_capacity_admission_is_serialized_before_spawn(rig):
    manager, _, _, process, _, _ = rig
    spawn = manager.spawn
    def checked_spawn(*args, **kwargs):
        # Another session cannot begin a same-device load during admission/spawn.
        other = claims.begin_operation('personal', 'warm', path=manager.ledger_path)
        try:
            assert other['ok'] is False and other['reason'] == 'capacity_pending'
        finally:
            if other['ok']:
                claims.finish_operation(other['operation_id'], path=manager.ledger_path)
        return spawn(*args, **kwargs)
    manager.spawn = checked_spawn
    result = load(manager)
    assert result['state'] == 'launch_accepted'
    # Once launched, the reservation remains, but unrelated admission is not
    # blocked for this service's entire lifetime.
    other = claims.begin_operation('personal', 'warm', path=manager.ledger_path)
    assert other['ok'] is True
    claims.finish_operation(other['operation_id'], path=manager.ledger_path)
    manager.stop('hermes-preview', result['operation_id'])


def test_no_operator_profile_refuses_without_launch(rig):
    manager, _, _, _, calls, _ = rig
    manager.profile_path = None
    assert load(manager)['reason'] == 'not_configured'
    assert calls == []


def test_process_wait_alone_does_not_establish_owned_exit(rig):
    manager, _, _, process, _, _ = rig
    process.stubborn = True
    process.wait = lambda timeout: 0
    result = load(manager)
    assert manager.stop('hermes-preview', result['operation_id'])['outcome'] == 'unknown'
    assert claims.list_claims(path=manager.ledger_path)
    process.code = -9


def test_mcp_validation_never_echoes_token_when_other_arguments_are_missing_or_invalid(rig, monkeypatch):
    import asyncio
    from vram_mcp import server
    manager, _, _, _, calls, _ = rig
    monkeypatch.setattr(server, '_kokoro_loader', manager)
    token = 'private-token-prefix-' + 'x' * 43
    async def scenario():
        for owner in (None, {'invalid': 'owner'}):
            arguments = {'auth_token': token, 'request_id': 'proof', 'connect_port': 12345}
            if owner is not None:
                arguments['owner'] = owner
            reply = await server.mcp.call_tool('load_kokoro', arguments)
            rendered = str(reply)
            assert 'private-token-prefix' not in rendered
            assert 'request_schema' in rendered
    asyncio.run(scenario())
    assert calls == []


def test_profile_size_bound_applies_before_reading_unbounded_input(rig, monkeypatch):
    import io
    from pathlib import Path
    _, _, profile, _, _, _ = rig
    opened = Path.open
    class BoundedFile(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 16385
            return super().read(size)
    def only_bounded(path, *args, **kwargs):
        if path == profile:
            return BoundedFile(b'x' * 16385)
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', only_bounded)
    with pytest.raises(kokoro_loader.Refusal, match='profile_bound'):
        kokoro_loader.read_profile(profile)


def test_retried_uncertain_spawn_keeps_capacity_and_never_claims_success(rig):
    manager, _, _, _, _, _ = rig
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise OSError('synthetic launch failure')
    manager.spawn = fail
    first = load(manager)
    assert first['outcome'] == 'unknown'
    assert load(manager) == first
    assert calls == [1]
    manager.refresh()
    assert manager.stop('hermes-preview', first['operation_id'])['outcome'] == 'unknown'
    assert claims.list_claims(path=manager.ledger_path)


def test_monitor_failure_prevents_spawn_and_preserves_failed_outcome(rig, monkeypatch):
    manager, _, _, _, calls, _ = rig
    manager._monitor = True
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic thread start failure')
    monkeypatch.setattr(kokoro_loader.threading.Thread, 'start', fail)
    first = load(manager)
    assert first['outcome'] == 'failed' and first['state'] == 'stopped'
    assert load(manager) == first
    assert calls == [] and claims.list_claims(path=manager.ledger_path) == []


def test_reservation_lost_during_spawn_settles_child_before_admission_releases(rig):
    manager, _, _, process, _, _ = rig
    spawn = manager.spawn
    def expire_during_spawn(*args, **kwargs):
        own = claims.list_claims(path=manager.ledger_path)[0]['claim_id']
        claims.release(own, path=manager.ledger_path)
        return spawn(*args, **kwargs)
    manager.spawn = expire_during_spawn
    assert load(manager)['outcome'] == 'failed'
    assert process.poll() is not None
    assert claims.list_coordination(path=manager.ledger_path)['operations'] == []


def test_unknown_spawn_with_lost_claim_retains_admission_until_server_exit(rig):
    manager, _, _, _, _, _ = rig
    def uncertain(*args, **kwargs):
        own = claims.list_claims(path=manager.ledger_path)[0]['claim_id']
        claims.release(own, path=manager.ledger_path)
        raise OSError('unknown spawn outcome')
    manager.spawn = uncertain
    result = load(manager)
    assert result['outcome'] == 'unknown'
    manager.refresh()
    other = claims.begin_operation('personal', 'warm', path=manager.ledger_path)
    try:
        assert other['ok'] is False and other['reason'] == 'capacity_pending'
    finally:
        if other['ok']:
            claims.finish_operation(other['operation_id'], path=manager.ledger_path)


def test_known_unsettled_child_keeps_startup_admission_until_verified_exit(rig):
    manager, _, _, process, _, _ = rig
    process.stubborn = True
    spawn = manager.spawn
    def expire_during_spawn(*args, **kwargs):
        own = claims.list_claims(path=manager.ledger_path)[0]['claim_id']
        claims.release(own, path=manager.ledger_path)
        return spawn(*args, **kwargs)
    manager.spawn = expire_during_spawn
    result = load(manager)
    assert result['outcome'] == 'unknown'
    other = claims.begin_operation('personal', 'warm', path=manager.ledger_path)
    try:
        assert other['ok'] is False and other['reason'] == 'capacity_pending'
    finally:
        if other['ok']:
            claims.finish_operation(other['operation_id'], path=manager.ledger_path)
    process.code = -9
    assert manager.status('hermes-preview', result['operation_id'])['state'] == 'stopped'
    assert claims.list_coordination(path=manager.ledger_path)['operations'] == []


def test_failed_admission_release_stays_unknown_until_retry_proves_release(rig, monkeypatch):
    manager, _, _, process, _, _ = rig
    finish = claims.finish_operation
    failing = True
    def unavailable(*args, **kwargs):
        if failing:
            raise OSError('synthetic admission release failure')
        return finish(*args, **kwargs)
    monkeypatch.setattr(claims, 'finish_operation', unavailable)
    result = load(manager)
    assert process.poll() is not None
    assert result['outcome'] == 'unknown' and result['state'] == 'cleanup_unknown'
    assert claims.list_coordination(path=manager.ledger_path)['operations']
    assert manager.status('hermes-preview', result['operation_id'])['outcome'] == 'unknown'
    failing = False
    recovered = manager.status('hermes-preview', result['operation_id'])
    assert recovered['state'] == 'stopped' and recovered['outcome'] == 'failed'
    assert claims.list_coordination(path=manager.ledger_path)['operations'] == []


def test_retained_admission_result_cannot_be_reported_as_released(rig, monkeypatch):
    manager, _, _, process, _, _ = rig
    monkeypatch.setattr(claims, 'finish_operation', lambda *args, **kwargs:
                        {'ok': True, 'retained': True})
    result = load(manager)
    assert result['outcome'] == 'unknown' and result['state'] == 'cleanup_unknown'
    assert process.poll() is not None
    assert claims.list_coordination(path=manager.ledger_path)['operations']


def test_status_without_matching_live_operation_cannot_trigger_its_cleanup(rig, monkeypatch):
    manager, _, _, process, _, _ = rig
    first = load(manager)
    renew = claims.renew
    monkeypatch.setattr(claims, 'renew', lambda *args, **kwargs: {'ok': False})
    assert manager.status('other-owner', first['operation_id'])['outcome'] == 'refused'
    assert process.poll() is None
    manager.stop('hermes-preview', first['operation_id'])
    second = Process()
    manager.spawn = lambda *args, **kwargs: second
    monkeypatch.setattr(claims, 'renew', renew)
    assert load(manager, 'request-two')['state'] == 'launch_accepted'
    monkeypatch.setattr(claims, 'renew', lambda *args, **kwargs: {'ok': False})
    assert manager.status('hermes-preview', first['operation_id'])['state'] == 'stopped'
    assert second.poll() is None


def test_prelaunch_admission_cleanup_error_requires_scoped_server_shutdown(rig, monkeypatch):
    manager, profile, _, _, calls, _ = rig
    manager.gpu_reading = lambda: Observation([{'index': 0, 'uuid': profile['gpu_uuid'],
                                               'free_mb': 1024}], 'fake')
    finish = claims.finish_operation
    def unavailable(*args, **kwargs):
        raise OSError('synthetic prelaunch admission release failure')
    monkeypatch.setattr(claims, 'finish_operation', unavailable)
    try:
        result = load(manager)
        assert result['outcome'] == 'unknown' and result['state'] == 'cleanup_unknown'
        assert result['reason'] == 'scoped_server_shutdown_required'
        assert result['operation_id'] is None and calls == []
        other = claims.begin_operation('personal', 'warm', path=manager.ledger_path)
        try:
            assert other['ok'] is False and other['reason'] == 'capacity_pending'
        finally:
            if other['ok']:
                finish(other['operation_id'], path=manager.ledger_path)
        assert load(manager, 'request-two')['reason'] == 'scoped_server_shutdown_required'
    finally:
        # Simulate scoped-server death releasing its private OS leases. This is
        # test cleanup, not a production reconciliation or successful receipt.
        for operation in claims.list_coordination(path=manager.ledger_path)['operations']:
            finish(operation['operation_id'], uncertain=True, path=manager.ledger_path)
