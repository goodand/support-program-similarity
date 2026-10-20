"""Run one Codex task through an enforced, tool-free request gateway.

The CLI reuses its existing login. This worker never reads authentication files.
Request filtering is separate from the CLI's read-only filesystem sandbox.
"""
import hashlib
import http.client
import http.server
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import ssl
import subprocess
import threading
import time


_UPSTREAM_HOST = 'chatgpt.com'
_UPSTREAM_PATH = '/backend-api/codex/responses'
_INSTRUCTIONS = 'Complete the supplied document similarity task and return the required JSON.'
_DISABLED_FEATURES = ('shell_tool', 'view_image', 'multi_agent', 'apps', 'hooks',
                      'skill_search', 'code_mode', 'code_mode_host', 'memories')
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_PREDICTION_BYTES = 16 * 1024 * 1024


class CodexWorkerError(RuntimeError):
    """An attempt failed; another backend is never used as a fallback."""


def _strict_object(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON object key')
            result[key] = value
        return result

    def constant(_value):
        raise ValueError('Non-finite JSON number')

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Non-finite JSON number')
        return number

    result = json.loads(text, object_pairs_hook=object_pairs,
                        parse_constant=constant, parse_float=finite_float)
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    return result


def _task_payload(prompt, schema, model, effort):
    # Construct a fresh allowlist. No CLI input, metadata, cookies, history,
    # previous_response_id, cache key, or client-selected schema can survive.
    return json.dumps({
        'model': model,
        'reasoning': {'effort': effort, 'summary': 'auto'},
        'instructions': _INSTRUCTIONS,
        'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}]}],
        'tools': [],
        'tool_choice': 'none',
        'text': {'format': {'type': 'json_schema', 'name': 'task_prediction',
                            'strict': True, 'schema': schema}},
        'stream': True,
        'store': False,
    }, ensure_ascii=False, allow_nan=False).encode('utf-8')


def _resolve_command(codex_executable):
    supplied = str(codex_executable) if codex_executable is not None else 'codex'
    resolved = shutil.which(supplied)
    if resolved is None and codex_executable is not None and Path(supplied).is_file():
        resolved = supplied
    if resolved is None:
        raise CodexWorkerError('Codex CLI was not found; an existing CLI login is required')
    executable = Path(resolved).resolve()
    suffix = executable.suffix.casefold()
    if suffix in ('.cmd', '.bat'):
        # Never invoke cmd.exe or interpolate a shell command. Resolve the
        # standard npm shim relative to its discovered location, not APPDATA.
        if executable.name.casefold() not in ('codex.cmd', 'codex.bat'):
            raise CodexWorkerError('Unsupported Codex shell launcher')
        executable = executable.parent / 'node_modules' / '@openai' / 'codex' / 'bin' / 'codex.js'
        if not executable.is_file():
            raise CodexWorkerError('The npm Codex shim has no adjacent codex.js; select the executable explicitly')
        suffix = '.js'
    if suffix in ('.js', '.mjs', '.cjs'):
        node = shutil.which('node')
        if node is None or Path(node).suffix.casefold() in ('.cmd', '.bat', '.ps1'):
            raise CodexWorkerError('A native Node executable is required for the Codex JavaScript launcher')
        return [str(Path(node).resolve()), str(executable)]
    if suffix == '.ps1':
        raise CodexWorkerError('PowerShell launchers are not executed; select a native Codex executable')
    return [str(executable)]


def _close_connection(connection):
    try:
        if connection.sock is not None:
            connection.sock.shutdown(socket.SHUT_RDWR)
    except (OSError, AttributeError):
        pass
    try:
        connection.close()
    except OSError:
        pass


class _GateState:
    def __init__(self, payload, deadline):
        self.payload = payload
        self.deadline = deadline
        self.lock = threading.Lock()
        self.connections = set()
        self.clients = set()
        self.secrets = set()
        self.errors = []
        self.requests = 0
        self.closed = False

    def remember_headers(self, headers):
        with self.lock:
            for name in ('Authorization', 'Cookie', 'Set-Cookie', 'ChatGPT-Account-ID'):
                for value in headers.get_all(name, []):
                    if value:
                        self.secrets.add(value)
                        if name == 'Authorization' and value.lower().startswith('bearer '):
                            self.secrets.add(value[7:].strip())

    def fail(self, reason):
        with self.lock:
            self.errors.append(reason)

    def attach(self, connection):
        with self.lock:
            if self.closed:
                raise RuntimeError('Gateway closed')
            self.connections.add(connection)
            self.requests += 1

    def detach(self, connection):
        with self.lock:
            self.connections.discard(connection)

    def close(self):
        with self.lock:
            self.closed = True
            connections = list(self.connections)
            clients = list(self.clients)
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for connection in connections:
            _close_connection(connection)

    def redact(self, data):
        text = data.decode('utf-8', errors='replace')
        with self.lock:
            secrets = sorted(filter(None, self.secrets), key=len, reverse=True)
        for value in secrets:
            text = text.replace(value, '[redacted]')
        # Also scrub credential-labelled CLI diagnostics before any file write.
        text = re.sub(r'(?im)^.*\b(?:authorization|(?:set-)?cookie|chatgpt-account-id)\b[\s\"\x27]*[:=].*$',
                      '[redacted credential header]', text)
        text = re.sub(r'(?i)\bbearer\s+[^\s\"\x27<>]+', 'Bearer [redacted]', text)
        return text.encode('utf-8')


class _GatewayServer(http.server.ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def handle_error(self, request, client_address):
        # Do not print exception text, incoming headers, or upstream bodies.
        self.gate.fail('gateway_handler_error')


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def setup(self):
        super().setup()
        gate = self.server.gate
        with gate.lock:
            gate.clients.add(self.connection)
            if gate.closed:
                self.connection.shutdown(socket.SHUT_RDWR)

    def finish(self):
        try:
            super().finish()
        finally:
            with self.server.gate.lock:
                self.server.gate.clients.discard(self.connection)

    def log_message(self, *args):
        pass

    def _reply(self, status):
        body = b'{"error":"Task gateway rejected the request or upstream response"}'
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _deny(self):
        self.server.gate.remember_headers(self.headers)
        self.server.gate.fail('endpoint_rejected')
        self._reply(403)

    do_GET = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _deny

    def do_POST(self):
        gate = self.server.gate
        gate.remember_headers(self.headers)
        if self.path != '/v1/responses':
            self._deny()
            return
        connection = None
        response = None
        started = False
        try:
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1 or self.headers.get('Transfer-Encoding'):
                raise ValueError('Invalid framing')
            length = int(lengths[0])
            if not 0 < length <= _MAX_REQUEST_BYTES:
                raise ValueError('Invalid length')
            if self.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                raise ValueError('Encoded request body')
            self.connection.settimeout(max(0.1, min(10, gate.deadline - time.monotonic())))
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('Incomplete request')
            _strict_object(body.decode('utf-8'))  # Validate, then discard ALL incoming JSON fields.
            authorizations = self.headers.get_all('Authorization', [])
            if (len(authorizations) != 1 or not authorizations[0].lower().startswith('bearer ')
                    or not authorizations[0][7:].strip()):
                raise ValueError('Missing CLI authorization')
            remaining = gate.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream',
                       'Accept-Encoding': 'identity', 'Authorization': authorizations[0]}
            # Cookie and arbitrary client headers are never forwarded. These
            # optional CLI headers support the existing ChatGPT login/protocol.
            for name in ('ChatGPT-Account-ID', 'OpenAI-Beta', 'User-Agent', 'originator'):
                values = self.headers.get_all(name, [])
                if len(values) > 1:
                    raise ValueError('Ambiguous header')
                if values:
                    headers[name] = values[0]
            connection = http.client.HTTPSConnection(_UPSTREAM_HOST, timeout=remaining,
                                                    context=ssl.create_default_context())
            gate.attach(connection)
            # HTTPSConnection neither follows redirects nor uses proxy env vars.
            # Neither the host nor path comes from the caller or CLI request.
            connection.request('POST', _UPSTREAM_PATH, body=gate.payload, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                gate.fail('upstream_http_' + str(int(response.status)))
                self._reply(502)
                return  # Never read, relay, or persist an upstream error body or Location.
            if response.getheader('Content-Encoding', 'identity').lower() != 'identity':
                gate.fail('upstream_encoding_rejected')
                self._reply(502)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            started = True
            while not gate.closed:
                chunk = response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (ValueError, UnicodeError, RecursionError):
            gate.fail('request_rejected')
            if not started:
                self._reply(400)
        except Exception:
            gate.fail('gateway_transport_error')
            if not started:
                try:
                    self._reply(502)
                except OSError:
                    pass
        finally:
            if connection is not None:
                _close_connection(connection)
                gate.detach(connection)
            if response is not None:
                response.close()
            self.close_connection = True


def _terminate_process_tree(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        # The npm launcher can own a native Codex child. Kill only this attempt's tree.
        try:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_attempt(prompt, schema, output_dir, *, model='gpt-6.1-sol', effort='low',
                timeout=2400, codex_executable=None):
    """Return a strict JSON prediction; domain validation belongs to the controller.

    Each attempt gets a new directory. CLI events, diagnostics, prediction, and
    request fingerprints remain there. No labels or prior conversation are read.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('prompt must be nonempty text')
    if not isinstance(schema, dict):
        raise ValueError('schema must be a JSON object')
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}', model):
        raise ValueError('Invalid model name')
    if effort not in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh'):
        raise ValueError('Invalid reasoning effort')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    schema_bytes = json.dumps(schema, ensure_ascii=False, allow_nan=False).encode('utf-8')
    frozen_schema = _strict_object(schema_bytes.decode('utf-8'))
    payload = _task_payload(prompt, frozen_schema, model, effort)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Attempt output directory already exists')
    command = _resolve_command(codex_executable)
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    worker_dir = output_dir / 'empty_worker_directory'
    worker_dir.mkdir(mode=0o700)
    schema_path = output_dir / 'output.schema.json'
    schema_path.write_bytes(schema_bytes)
    prediction_path = output_dir / 'prediction.json'
    start = time.monotonic()
    gate = _GateState(payload, start + timeout)
    try:
        server = _GatewayServer(('127.0.0.1', 0), _Handler)
    except OSError:
        raise CodexWorkerError('The loopback task gateway could not start') from None
    server.gate = gate
    server_thread = threading.Thread(target=server.serve_forever,
                                     kwargs={'poll_interval': 0.05}, daemon=True)
    try:
        server_thread.start()
    except RuntimeError:
        server.server_close()
        raise CodexWorkerError('The task gateway thread could not start') from None
    base_url = 'http://127.0.0.1:' + str(server.server_address[1]) + '/v1'
    args = command + ['exec', '--ignore-user-config', '--skip-git-repo-check', '--ephemeral',
                      '-C', str(worker_dir), '-s', 'read-only', '-m', model]
    configs = {'model_reasoning_effort': effort, 'model_provider': 'task_gate',
               'model_providers.task_gate.name': 'Task-only gateway',
               'model_providers.task_gate.base_url': base_url,
               'model_providers.task_gate.wire_api': 'responses',
               'model_providers.task_gate.requires_openai_auth': True,
               'model_providers.task_gate.supports_websockets': False,
               'web_search': 'disabled'}
    for key, value in configs.items():
        args.extend(['-c', key + '=' + json.dumps(value)])
    for feature in _DISABLED_FEATURES:
        args.extend(['--disable', feature])
    args.extend(['--output-schema', str(schema_path), '-o', str(prediction_path), '--json', '-'])
    process = None
    stdout = stderr = b''
    failure = None
    try:
        platform_options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt'
                            else {'start_new_session': True})
        process = subprocess.Popen(args, cwd=worker_dir, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   shell=False, **platform_options)
        try:
            stdout, stderr = process.communicate(b'Execute the supplied structured task.',
                                                timeout=max(0.1, gate.deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            failure = 'Codex worker timed out'
            _terminate_process_tree(process)
            stdout, stderr = process.communicate(timeout=10)
        if failure is None and process.returncode != 0:
            failure = 'Codex worker failed with exit code ' + str(process.returncode)
    except (OSError, subprocess.SubprocessError):
        failure = 'Codex worker could not complete its process'
    finally:
        try:
            if process is not None and process.poll() is None:
                _terminate_process_tree(process)
                process.wait(timeout=10)
        finally:
            gate.close()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
    (output_dir / 'worker.stdout.log').write_bytes(gate.redact(stdout))
    (output_dir / 'worker.stderr.log').write_bytes(gate.redact(stderr))
    metadata = {'model': model, 'effort': effort, 'reasoning_summary': 'auto',
                'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                'schema_sha256': hashlib.sha256(schema_bytes).hexdigest(),
                'request_sha256': hashlib.sha256(payload).hexdigest(),
                'upstream': 'https://' + _UPSTREAM_HOST + _UPSTREAM_PATH,
                'upstream_requests': gate.requests, 'gateway_errors': gate.errors,
                'returncode': process.returncode if process is not None else None,
                'elapsed_seconds': time.monotonic() - start, 'process_error': failure}
    (output_dir / 'execution.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if failure is not None:
        raise CodexWorkerError(failure) from None
    if gate.errors or not gate.requests:
        raise CodexWorkerError('The task gateway did not complete an authorized inference request')
    try:
        if prediction_path.stat().st_size > _MAX_PREDICTION_BYTES:
            raise ValueError('Prediction too large')
        prediction = _strict_object(prediction_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError, RecursionError):
        raise CodexWorkerError('Codex did not produce a strict JSON prediction object') from None
    return prediction
