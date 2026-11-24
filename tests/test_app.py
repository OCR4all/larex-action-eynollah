# SPDX-License-Identifier: MIT
import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from larex_actions.verifier import canonical_dispatch_request
from pydantic import ValidationError

from larex_action.app import create_app, worker_settings


def encode(value):
    return base64.urlsafe_b64encode(value).rstrip(b'=').decode()


def signed_preflight(path, secret):
    body = json.dumps(
        {
            'protocolVersion': 1,
            'requestId': 'preflight-1',
            'processorId': 'eynollah-segmentation',
            'capabilities': {'incrementalPageResults': True},
        }
    ).encode()
    timestamp = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    body_hash = encode(hashlib.sha256(body).digest())
    canonical = canonical_dispatch_request(
        method='POST',
        path_and_query=path,
        run_id='preflight-1',
        processor_id='eynollah-segmentation',
        timestamp=timestamp,
        nonce='nonce-1',
        body_hash=body_hash,
    )
    return body, {
        'X-LAREX-Action-Auth': 'hmac-sha256;v=1',
        'X-LAREX-Action-Processor': 'eynollah-segmentation',
        'X-LAREX-Action-Run-Id': 'preflight-1',
        'X-LAREX-Action-Timestamp': timestamp,
        'X-LAREX-Action-Nonce': 'nonce-1',
        'X-LAREX-Action-Body-SHA256': body_hash,
        'X-LAREX-Action-Signature': 'v1=' + encode(hmac.new(secret.encode(), canonical.encode(), 'sha256').digest()),
    }


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings(LAREX_ROUTE_PREFIXES=['/eynollah']))) as client:
        yield client


@pytest.mark.parametrize('path', ['/health', '/eynollah/health'])
def test_health(client, path):
    assert client.get(path).json() == {'status': 'ok'}


def test_preflight(client):
    body, headers = signed_preflight('/eynollah/preflight', 'test-dispatch-secret')
    response = client.post('/eynollah/preflight', content=body, headers=headers)
    assert response.status_code == 200
    assert response.json()['capabilities']['incrementalPageResults']


def test_preflight_rejects_other_secrets(client):
    body, headers = signed_preflight('/preflight', 'other-secret')
    assert client.post('/preflight', content=body, headers=headers).status_code == 401


def test_unsigned_dispatch_is_rejected(client):
    assert client.post('/dispatch', json={}).status_code == 401


def test_threads_are_divided_between_runs(settings, monkeypatch):
    monkeypatch.setattr('larex_action.app.available_cpus', lambda: 8)
    assert worker_settings(settings()).threads == 8
    assert worker_settings(settings(ACTION_MAX_CONCURRENT_RUNS=3)).threads == 2
    assert worker_settings(settings(EYNOLLAH_CPU_THREADS=5)).threads == 5


def test_empty_environment_variables_use_the_default(settings, monkeypatch):
    monkeypatch.setenv('EYNOLLAH_PAGE_WORKERS', '')
    monkeypatch.setenv('ACTION_PROCESSING_TIMEOUT', 'None')
    assert settings().EYNOLLAH_PAGE_WORKERS is None
    assert settings().ACTION_PROCESSING_TIMEOUT is None


@pytest.mark.parametrize('device', ['gpu', 'cuda:', 'CPU', 'cuda:a'])
def test_invalid_device(settings, device):
    with pytest.raises(ValidationError):
        settings(EYNOLLAH_DEVICE=device)
