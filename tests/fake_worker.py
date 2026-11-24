# SPDX-License-Identifier: MIT
import os
import threading
import time
from dataclasses import replace

from larex_action.worker import LoadRequest, Reply, Started

BARRIER_SECONDS = 0.3


def fake_worker(connection, settings):
    if settings.device == 'unavailable':
        connection.send(Started(error='CUDA is unavailable'))
        return
    connection.send(Started(device='cpu', page_workers=settings.page_workers or 1))
    lock = threading.Lock()

    def answer(reply):
        with lock:
            connection.send(reply)

    try:
        while (request := connection.recv()) is not None:
            threading.Thread(target=handle, args=(request, settings, answer), daemon=True).start()
    except (EOFError, OSError):
        pass
    os._exit(0)


def handle(request, settings, answer):
    if isinstance(request, LoadRequest):
        if settings.device == 'hang':
            time.sleep(60)
        if settings.device == 'crash':
            os._exit(7)
        failed = settings.device == 'broken'
        answer(Reply(request.id, 'failed' if failed else 'ok', 'The models are damaged' if failed else ''))
        return
    behaviour = request.image.read_text()
    reply = Reply(request.id, 'ok', xml=f'<PcGts pid="{os.getpid()}"/>'.encode(), seconds=0.1)
    if behaviour == 'crash':
        time.sleep(BARRIER_SECONDS)
        os._exit(1)
    if behaviour == 'hang':
        time.sleep(60)
    if behaviour == 'slow':
        time.sleep(BARRIER_SECONDS)
    if behaviour == 'failed':
        reply = replace(reply, status='failed', message='The image format is not supported', xml=None)
    if behaviour == 'broken':
        reply = replace(reply, status='broken', message='out of memory', xml=None)
    if behaviour == 'empty':
        reply = replace(reply, status='empty', xml=None)
    answer(reply)
