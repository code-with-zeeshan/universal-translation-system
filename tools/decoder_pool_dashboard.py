from flask import Flask, render_template_string
import json
import requests
import os

app = Flask(__name__)

TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Decoder Pool Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background: #f0f0f0; }
        .healthy { color: green; }
        .unhealthy { color: red; }
    </style>
</head>
<body>
    <h1>Decoder Pool Dashboard</h1>
    <table>
        <tr>
            <th>Node ID</th>
            <th>Endpoint</th>
            <th>Region</th>
            <th>GPU</th>
            <th>Capacity</th>
            <th>Status</th>
        </tr>
        {% for node in nodes %}
        <tr>
            <td>{{ node['node_id'] }}</td>
            <td><a href="{{ node['endpoint'] }}" target="_blank">{{ node['endpoint'] }}</a></td>
            <td>{{ node['region'] }}</td>
            <td>{{ node['gpu_type'] }}</td>
            <td>{{ node['capacity'] }}</td>
            <td class="{{ 'healthy' if node['healthy'] else 'unhealthy' }}">{{ 'Healthy' if node['healthy'] else 'Unhealthy' }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
'''

def load_pool():
    pool_path = os.path.join("configs", "decoder_pool.json")
    if os.path.exists(pool_path):
        with open(pool_path, "r") as f:
            return json.load(f)
    return []

def check_health(endpoint):
    try:
        resp = requests.get(f"{endpoint}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False

@app.route("/")
def dashboard():
    nodes = load_pool()
    for node in nodes:
        node['healthy'] = check_health(node['endpoint'])
    return render_template_string(TEMPLATE, nodes=nodes)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True) 