"""Synthetic gateway and process tests: no Codex or remote API is invoked."""
import io
import json
from pathlib import Path
import ssl
import subprocess
import tempfile
import threading
import unittest
from unittest import mock
from urllib.parse import urlsplit

from support_program_similarity import codex_worker as worker


SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string'}},
          'required': ['answer'], 'additionalProperties': False}
PROMPT = 'Use only this synthetic task.'
TOKEN = 'synthetic-worker-token-never-save'
ACCOUNT = 'synthetic-worker-account-never-save'
COOKIE = 'synthetic-cookie-never-save'


class FakeResponse(io.BytesIO):
    def __init__(self, status=200):
        super().__init__(b'data: {"type":"response.completed"}\n\n' if status == 200
                         else b'PRIVATE UPSTREAM ERROR BODY MUST NOT ESCAPE')
        self.status = status
        self.read_calls = 0

    def getheader(self, name, default=None):
        return {'Content-Encoding': 'identity', 'Content-Type': 'text/event-stream',
                'Location': 'https://unapproved.invalid/steal'}.get(name, default)

    def read1(self, size):
        self.read_calls += 1
        return super().read1(size)


class FakeHTTPSConnection:
    def __init__(self, response):
        self.response = response
        self.sock = None
        self.calls = []
        self.closed = False

    def request(self, method, path, *, body, headers):
        self.calls.append((method, path, body, headers))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class MemorySocket:
    def __init__(self, request):
        self.request = io.BytesIO(request)
        self.response = bytearray()

    def makefile(self, mode, buffering=-1):
        return self.request

    def sendall(self, data):
        self.response.extend(data)

    def settimeout(self, timeout):
        pass

    def shutdown(self, how):
        pass


class MemoryGateway:
    """Exercise the actual HTTP parser/handler without opening a socket."""
    def __init__(self, address, handler):
        assert address == ('127.0.0.1', 0)
        self.server_address = ('127.0.0.1', 32123)
        self.handler = handler
        self.event = threading.Event()
        self.shutdown_called = self.close_called = self.stopped = False

    def serve_forever(self, poll_interval):
        self.event.wait()
        self.stopped = True

    def shutdown(self):
        self.shutdown_called = True
        self.event.set()

    def server_close(self):
        self.close_called = True

    def post(self, path, body, headers):
        lines = ['POST ' + path + ' HTTP/1.1', 'Host: 127.0.0.1',
                 'Content-Length: ' + str(len(body)),
                 *[key + ': ' + value for key, value in headers.items()], '', '']
        connection = MemorySocket('\r\n'.join(lines).encode('ascii') + body)
        self.handler(connection, ('127.0.0.1', 12345), self)
        head, reply = bytes(connection.response).split(b'\r\n\r\n', 1)
        return int(head.split(b' ', 2)[1]), reply


class FakeProcess:
    def __init__(self, args, *, returncode=0, prediction=b'{"answer":"synthetic"}',
                 endpoint=None, timeout_first=False, gateway):
        self.args = args
        self.result_code = returncode
        self.returncode = None
        self.prediction = prediction
        self.endpoint = endpoint
        self.timeout_first = timeout_first
        self.gateway = gateway
        self.communications = 0
        self.pid = 987654321
        self.stdin = self.stdout = self.stderr = None
        self.reply = None
        self.gateway_port = None

    def communicate(self, input=None, timeout=None):
        self.communications += 1
        if self.timeout_first and self.communications == 1:
            raise subprocess.TimeoutExpired(self.args, timeout)
        if self.communications == 1:
            configs = [self.args[i + 1] for i, value in enumerate(self.args[:-1]) if value == '-c']
            url = next(json.loads(c.split('=', 1)[1]) for c in configs
                       if c.startswith('model_providers.task_gate.base_url='))
            parsed = urlsplit(url)
            self.gateway_port = parsed.port
            request = {'model': 'unwanted-model', 'reasoning': {'effort': 'high'},
                       'instructions': 'unwanted CLI instructions', 'input': [{'text': 'PRIVATE CLI HISTORY'}],
                       'tools': [{'type': 'shell'}], 'tool_choice': 'auto', 'store': True,
                       'text': {'format': {'schema': {'UNWANTED_SCHEMA': True}}},
                       'previous_response_id': 'prior-response', 'client_metadata': {'cookie': COOKIE},
                       'include': ['private-context'], 'prompt_cache_key': 'private-cache-key',
                       'Authorization': TOKEN, 'parallel_tool_calls': True}
            self.reply = self.gateway.post(self.endpoint or parsed.path + '/responses',
                                           json.dumps(request).encode('utf-8'),
                                           {'Authorization': 'Bearer ' + TOKEN,
                                            'ChatGPT-Account-ID': ACCOUNT, 'Cookie': COOKIE,
                                            'X-CLI-Context': 'PRIVATE HEADER CONTEXT'})
            path = Path(self.args[self.args.index('-o') + 1])
            path.write_bytes(self.prediction)
        self.returncode = self.result_code if not self.timeout_first else -9
        stdout = ('Authorization: Bearer ' + TOKEN + '\nCookie: ' + COOKIE + '\n' + ACCOUNT).encode()
        return stdout, ('synthetic diagnostic ' + TOKEN).encode()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class CodexWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.codex_worker_test_',
                                               dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_mock(self, *, status=200, process_options=None, launch_error=False):
        response = FakeResponse(status)
        upstream = FakeHTTPSConnection(response)
        processes = []

        def launch(args, **kwargs):
            self.assertFalse(kwargs['shell'])
            self.assertIn('--ignore-user-config', args)
            self.assertIn('--ephemeral', args)
            self.assertEqual(args[args.index('-s') + 1], 'read-only')
            self.assertNotIn('--dangerously-bypass-approvals-and-sandbox', args)
            self.assertNotIn('--yolo', args)
            if launch_error:
                raise OSError('PRIVATE STARTUP ERROR ' + TOKEN)
            process = FakeProcess(args, gateway=self.gateway, **(process_options or {}))
            processes.append(process)
            return process

        def serve(address, handler):
            self.gateway = MemoryGateway(address, handler)
            return self.gateway

        self.upstream = upstream
        self.response = response
        self.processes = processes
        self.output = self.root / 'attempt'
        with mock.patch.object(worker, '_resolve_command', return_value=['codex-fixture']), \
             mock.patch.object(worker, '_GatewayServer', side_effect=serve), \
             mock.patch.object(worker.subprocess, 'Popen', side_effect=launch) as popen, \
             mock.patch.object(worker.http.client, 'HTTPSConnection', return_value=upstream) as https:
            self.popen = popen
            self.https = https
            return worker.run_attempt(PROMPT, SCHEMA, self.output,
                                      model='gpt-6.1-sol', effort='low', timeout=5)

    def test_gateway_forces_task_model_effort_schema_and_excludes_cli_context(self):
        self.assertEqual(self.run_mock(), {'answer': 'synthetic'})
        self.assertEqual(self.https.call_count, 1)
        self.assertEqual(self.https.call_args.args, ('chatgpt.com',))
        context = self.https.call_args.kwargs['context']
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        method, path, data, headers = self.upstream.calls[0]
        self.assertEqual((method, path), ('POST', '/backend-api/codex/responses'))
        payload = json.loads(data)
        self.assertEqual(set(payload), {'model', 'reasoning', 'instructions', 'input', 'tools',
                                       'tool_choice', 'text', 'stream', 'store'})
        self.assertEqual(payload['model'], 'gpt-6.1-sol')
        self.assertEqual(payload['reasoning'], {'effort': 'low', 'summary': 'auto'})
        self.assertEqual(payload['instructions'], worker._INSTRUCTIONS)
        self.assertEqual(payload['input'], [{'role': 'user', 'content': [{'type': 'input_text', 'text': PROMPT}]}])
        self.assertEqual(payload['tools'], [])
        self.assertEqual(payload['tool_choice'], 'none')
        self.assertFalse(payload['store'])
        self.assertTrue(payload['stream'])
        self.assertEqual(payload['text']['format']['schema'], SCHEMA)
        self.assertTrue(payload['text']['format']['strict'])
        for prohibited in ('PRIVATE CLI HISTORY', 'UNWANTED_SCHEMA', TOKEN, ACCOUNT, COOKIE,
                           'prior-response', 'private-cache-key', 'PRIVATE HEADER CONTEXT'):
            self.assertNotIn(prohibited, data.decode('utf-8'))
        self.assertEqual(headers['Authorization'], 'Bearer ' + TOKEN)
        self.assertEqual(headers['ChatGPT-Account-ID'], ACCOUNT)
        self.assertNotIn('Cookie', headers)
        self.assertNotIn('X-CLI-Context', headers)
        self.assertTrue(self.upstream.closed)
        for path in self.output.rglob('*'):
            if path.is_file():
                saved = path.read_bytes()
                for secret in (TOKEN, ACCOUNT, COOKIE):
                    self.assertNotIn(secret.encode(), saved, path.name)
        self.assertTrue(self.gateway.shutdown_called)
        self.assertTrue(self.gateway.close_called)
        self.assertTrue(self.gateway.stopped)

    def test_rejects_non_allowlisted_endpoint_without_upstream_call(self):
        with self.assertRaises(worker.CodexWorkerError):
            self.run_mock(process_options={'endpoint': '/other/responses'})
        self.assertEqual(self.https.call_count, 0)
        self.assertEqual(self.processes[0].reply[0], 403)
        self.assertEqual(self.popen.call_count, 1)

    def test_redirect_and_upstream_error_body_never_leave_gateway(self):
        with self.assertRaises(worker.CodexWorkerError):
            self.run_mock(status=302)
        self.assertEqual(self.https.call_count, 1)
        self.assertEqual(len(self.upstream.calls), 1)
        self.assertEqual(self.response.read_calls, 0)
        self.assertEqual(self.processes[0].reply[0], 502)
        self.assertNotIn(b'PRIVATE UPSTREAM ERROR BODY', self.processes[0].reply[1])
        for path in self.output.rglob('*'):
            if path.is_file():
                self.assertNotIn(b'PRIVATE UPSTREAM ERROR BODY', path.read_bytes())

    def test_nonzero_exit_does_not_return_prediction_or_fallback(self):
        with self.assertRaisesRegex(worker.CodexWorkerError, 'exit code 7'):
            self.run_mock(process_options={'returncode': 7})
        self.assertEqual(self.popen.call_count, 1)
        self.assertTrue(self.upstream.closed)

    def test_timeout_terminates_attempt_and_has_no_fallback(self):
        def stop(process):
            process.returncode = -9

        with mock.patch.object(worker, '_terminate_process_tree', side_effect=stop) as terminate:
            with self.assertRaisesRegex(worker.CodexWorkerError, 'timed out'):
                self.run_mock(process_options={'timeout_first': True})
        terminate.assert_called_once_with(self.processes[0])
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(self.https.call_count, 0)

    def test_launch_error_is_sanitized_and_gateway_is_closed(self):
        with self.assertRaisesRegex(worker.CodexWorkerError, 'could not complete'):
            self.run_mock(launch_error=True)
        self.assertEqual(self.popen.call_count, 1)
        self.assertEqual(self.https.call_count, 0)
        self.assertTrue(self.gateway.shutdown_called)
        self.assertTrue(self.gateway.close_called)
        self.assertTrue(self.gateway.stopped)
        for path in self.output.rglob('*'):
            if path.is_file():
                self.assertNotIn(TOKEN.encode(), path.read_bytes())

    def test_strict_prediction_and_existing_output_directory(self):
        with self.assertRaisesRegex(worker.CodexWorkerError, 'strict JSON'):
            self.run_mock(process_options={'prediction': b'{"answer":"a","answer":"b"}'})
        with mock.patch.object(worker, '_resolve_command') as resolve:
            with self.assertRaises(FileExistsError):
                worker.run_attempt(PROMPT, SCHEMA, self.output)
            resolve.assert_not_called()
        for invalid in ('[]', '{"v":NaN}', '{"v":Infinity}', '{"v":1e999}',
                        '{"inner":{"x":1,"x":2}}', '```json\n{}\n```'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    worker._strict_object(invalid)

    def test_windows_npm_shim_resolves_without_a_shell_or_appdata(self):
        directory = self.root / 'npm path with spaces & metacharacters'
        directory.mkdir()
        shim = directory / 'codex.cmd'
        shim.write_text('synthetic shim; never executed', encoding='utf-8')
        script = directory / 'node_modules' / '@openai' / 'codex' / 'bin' / 'codex.js'
        script.parent.mkdir(parents=True)
        script.write_text('// synthetic; never executed', encoding='utf-8')
        node = self.root / 'node.exe'
        node.write_text('synthetic; never executed', encoding='utf-8')
        def which(name):
            return str(node) if name == 'node' else str(shim)
        with mock.patch.object(worker.shutil, 'which', side_effect=which):
            self.assertEqual(worker._resolve_command(None), [str(node), str(script)])
        script.unlink()
        with mock.patch.object(worker.shutil, 'which', side_effect=which):
            with self.assertRaises(worker.CodexWorkerError):
                worker._resolve_command(None)


if __name__ == '__main__':
    unittest.main()
