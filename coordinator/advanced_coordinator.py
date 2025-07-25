import threading
import time
import json
import requests
from flask import Flask, request, jsonify, render_template_string, redirect, url_for, session
import random
import os
from prometheus_client import start_http_server, Counter, Gauge
from functools import wraps
from datetime import datetime
from flask_smorest import Api, Blueprint
from flask.views import MethodView
from marshmallow import Schema, fields
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from opentelemetry import trace
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# OpenTelemetry setup
trace.set_tracer_provider(
    TracerProvider(resource=Resource.create({SERVICE_NAME: "coordinator-service"}))
)
otlp_exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
tracer = trace.get_tracer(__name__)

POOL_PATH = os.path.join("configs", "decoder_pool.json")
CONFIG_PATH = os.path.join("config", "coordinator_config.json")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
AUTH_TOKEN = os.environ.get("COORDINATOR_TOKEN", "changeme123")
SECRET_KEY = os.environ.get("COORDINATOR_SECRET", "supersecret")

app = Flask(__name__)
app.secret_key = SECRET_KEY
FlaskInstrumentor().instrument_app(app)
api = Api(app)
blp = Blueprint('api', 'api', url_prefix='/api', description='Coordinator API')

# OpenAPI schema for status
class DecoderNodeSchema(Schema):
    node_id = fields.Str()
    endpoint = fields.Str()
    region = fields.Str()
    gpu_type = fields.Str()
    capacity = fields.Int()
    healthy = fields.Bool()
    load = fields.Int()
    uptime = fields.Int()

class StatusSchema(Schema):
    model_version = fields.Str()
    decoders = fields.List(fields.Nested(DecoderNodeSchema))

# JWT authentication
import jwt
JWT_SECRET = os.environ.get("COORDINATOR_JWT_SECRET", "jwtsecret123")
def require_jwt(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return {"error": "Unauthorized"}, 401
        token = auth_header.replace('Bearer ', '')
        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except Exception:
            return {"error": "Invalid token"}, 401
        return f(*args, **kwargs)
    return decorated

# Live config reload
class ConfigReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(CONFIG_PATH):
            with tracer.start_as_current_span("config_reload"):
                print("[Coordinator] Config file changed, reloading...")
                # Reload config logic here
                # (For demo, just print. In production, update in-memory config.)

observer = Observer()
observer.schedule(ConfigReloadHandler(), path=os.path.dirname(CONFIG_PATH) or '.', recursive=False)
observer.start()

# Prometheus metrics
requests_total = Counter('coordinator_requests_total', 'Total requests received', ['endpoint'])
requests_errors = Counter('coordinator_requests_errors', 'Total errors', ['endpoint'])
decoder_active = Gauge('coordinator_decoder_active', 'Active decoders')
decoder_load = Gauge('coordinator_decoder_load', 'Current load per decoder', ['node_id'])
decoder_uptime = Gauge('coordinator_decoder_uptime', 'Uptime (s) per decoder', ['node_id'])

# In-memory stats for analytics
decoder_stats = {}

TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Coordinator Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background: #f0f0f0; }
        .healthy { color: green; }
        .unhealthy { color: red; }
        .admin { margin-bottom: 1em; }
        .chart-container { width: 100%; max-width: 900px; margin: 2em auto; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Coordinator Dashboard</h1>
    {% if not session.get('logged_in') %}
    <form method="post" action="/login" class="admin">
        <input type="password" name="token" placeholder="Admin Token" />
        <button type="submit">Login</button>
    </form>
    {% else %}
    <form method="post" action="/logout" class="admin">
        <button type="submit">Logout</button>
    </form>
    {% endif %}
    <div class="chart-container">
        <canvas id="loadChart"></canvas>
    </div>
    <div class="chart-container">
        <canvas id="uptimeChart"></canvas>
    </div>
    <table>
        <tr>
            <th>Node ID</th>
            <th>Endpoint</th>
            <th>Region</th>
            <th>GPU</th>
            <th>Capacity</th>
            <th>Current Load</th>
            <th>Uptime (s)</th>
            <th>Status</th>
            <th>Route</th>
        </tr>
        {% for node in nodes %}
        <tr>
            <td>{{ node['node_id'] }}</td>
            <td><a href="{{ node['endpoint'] }}" target="_blank">{{ node['endpoint'] }}</a></td>
            <td>{{ node['region'] }}</td>
            <td>{{ node['gpu_type'] }}</td>
            <td>{{ node['capacity'] }}</td>
            <td>{{ node['load'] }}</td>
            <td>{{ node['uptime'] }}</td>
            <td class="{{ 'healthy' if node['healthy'] else 'unhealthy' }}">{{ 'Healthy' if node['healthy'] else 'Unhealthy' }}</td>
            <td>
                {% if node['healthy'] and session.get('logged_in') %}
                <form method="post" action="/manual_route">
                    <input type="hidden" name="node_id" value="{{ node['node_id'] }}" />
                    <input type="text" name="text" placeholder="Text to translate" />
                    <input type="text" name="source_lang" placeholder="Source" size="4" />
                    <input type="text" name="target_lang" placeholder="Target" size="4" />
                    <button type="submit">Route</button>
                </form>
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>
    <script>
        const loadData = {{ loads|safe }};
        const uptimeData = {{ uptimes|safe }};
        const nodeLabels = {{ node_labels|safe }};
        new Chart(document.getElementById('loadChart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: nodeLabels,
                datasets: [{
                    label: 'Current Load',
                    data: loadData,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                }]
            },
            options: { responsive: true, plugins: { legend: { display: false } } }
        });
        new Chart(document.getElementById('uptimeChart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: nodeLabels,
                datasets: [{
                    label: 'Uptime (s)',
                    data: uptimeData,
                    backgroundColor: 'rgba(75, 192, 192, 0.5)'
                }]
            },
            options: { responsive: true, plugins: { legend: { display: false } } }
        });
    </script>
</body>
</html>
'''

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated

class DecoderPool:
    def __init__(self, pool_path=POOL_PATH):
        self.pool_path = pool_path
        self.lock = threading.Lock()
        self.reload()

    def reload(self):
        with self.lock:
            if os.path.exists(self.pool_path):
                with open(self.pool_path, "r") as f:
                    self.pool = json.load(f)
            else:
                self.pool = []
            for node in self.pool:
                node['healthy'] = self.check_health(node['endpoint'])
                node['load'] = self.get_load(node)
                node['uptime'] = self.get_uptime(node)

    def get_healthy_decoders(self):
        with self.lock:
            return [n for n in self.pool if n.get('healthy')]

    def check_health(self, endpoint):
        try:
            resp = requests.get(f"{endpoint}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def get_load(self, node):
        try:
            resp = requests.get(f"{node['endpoint']}/metrics", timeout=2)
            for line in resp.text.splitlines():
                if 'active_connections' in line:
                    return int(float(line.split()[-1]))
        except Exception:
            pass
        return 0

    def get_uptime(self, node):
        try:
            resp = requests.get(f"{node['endpoint']}/metrics", timeout=2)
            for line in resp.text.splitlines():
                if 'process_uptime_seconds' in line:
                    return int(float(line.split()[-1]))
        except Exception:
            pass
        return 0

    def add_decoder(self, node):
        with self.lock:
            self.pool.append(node)
            self.save()

    def remove_decoder(self, node_id):
        with self.lock:
            self.pool = [n for n in self.pool if n['node_id'] != node_id]
            self.save()

    def save(self):
        with open(self.pool_path, "w") as f:
            json.dump(self.pool, f, indent=2)

    def pick_least_loaded(self):
        healthy = self.get_healthy_decoders()
        if not healthy:
            return None
        node = min(healthy, key=lambda n: n.get('load', 0))
        return node

pool = DecoderPool()

# Track request times for analytics
def background_health_check():
    while True:
        pool.reload()
        healthy = pool.get_healthy_decoders()
        decoder_active.set(len(healthy))
        for node in healthy:
            decoder_load.labels(node_id=node['node_id']).set(node.get('load', 0))
            decoder_uptime.labels(node_id=node['node_id']).set(node.get('uptime', 0))
        time.sleep(10)

@blp.route('/status')
class StatusResource(MethodView):
    @blp.response(200, StatusSchema)
    def get(self):
        with tracer.start_as_current_span("status_endpoint") as span:
            span.set_attribute("model_version", MODEL_VERSION)
            # pool is assumed to be globally available
            return {"model_version": MODEL_VERSION, "decoders": pool.get_healthy_decoders()}

@blp.route('/add_decoder')
class AddDecoderResource(MethodView):
    @require_jwt
    def post(self):
        with tracer.start_as_current_span("add_decoder") as span:
            node = request.json
            span.set_attribute("node_id", node.get('node_id', ''))
            node['healthy'] = pool.check_health(node['endpoint'])
            node['load'] = pool.get_load(node)
            node['uptime'] = pool.get_uptime(node)
            pool.add_decoder(node)
            return {"status": "added", "node": node}

@blp.route('/remove_decoder')
class RemoveDecoderResource(MethodView):
    @require_jwt
    def post(self):
        with tracer.start_as_current_span("remove_decoder") as span:
            node_id = request.json.get('node_id')
            span.set_attribute("node_id", node_id)
            pool.remove_decoder(node_id)
            return {"status": "removed", "node_id": node_id}

@blp.route('/decode')
class DecodeResource(MethodView):
    def post(self):
        with tracer.start_as_current_span("decode_proxy") as span:
            node = pool.pick_least_loaded()
            if not node:
                span.set_attribute("error", True)
                return {"error": "No healthy decoders available"}, 503
            span.set_attribute("node_id", node['node_id'])
            try:
                headers = dict(request.headers)
                resp = requests.post(f"{node['endpoint']}/decode", data=request.data, headers=headers, timeout=10)
                return (resp.content, resp.status_code, resp.headers.items())
            except Exception as e:
                span.set_attribute("error", True)
                return {"error": f"Failed to contact decoder: {e}"}, 502

api.register_blueprint(blp)

@app.route("/dashboard")
def dashboard():
    pool.reload()
    node_labels = [n['node_id'] for n in pool.pool]
    loads = [n.get('load', 0) for n in pool.pool]
    uptimes = [n.get('uptime', 0) for n in pool.pool]
    return render_template_string(TEMPLATE, nodes=pool.pool, node_labels=node_labels, loads=loads, uptimes=uptimes, session=session)

@app.route("/manual_route", methods=["POST"])
@require_auth
def manual_route():
    node_id = request.form['node_id']
    text = request.form['text']
    source_lang = request.form['source_lang']
    target_lang = request.form['target_lang']
    node = next((n for n in pool.pool if n['node_id'] == node_id), None)
    if not node:
        return "Node not found", 404
    try:
        headers = {'X-Target-Language': target_lang, 'Content-Type': 'application/json'}
        data = json.dumps({'text': text, 'source_lang': source_lang, 'target_lang': target_lang})
        resp = requests.post(f"{node['endpoint']}/decode", data=data, headers=headers, timeout=10)
        return resp.text
    except Exception as e:
        return f"Error: {e}", 502

@app.route("/metrics")
def metrics():
    from prometheus_client import generate_latest
    return generate_latest(), 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}

@app.route("/login", methods=["POST"])
def login():
    token = request.form.get('token')
    if token == AUTH_TOKEN:
        session['logged_in'] = True
    return redirect(url_for('dashboard'))

@app.route("/logout", methods=["POST"])
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('dashboard'))

if __name__ == "__main__":
    start_http_server(9200)  # Prometheus metrics on 9200
    threading.Thread(target=background_health_check, daemon=True).start()
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5100, debug=debug_mode)
