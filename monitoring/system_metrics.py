# monitoring/system_metrics.py
import time
import psutil
from prometheus_client import start_http_server, Gauge
try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# vocabulary metrics collection
try:
    from metrics_collector import collect_vocabulary_metrics
    HAS_VOCAB_METRICS = True
except ImportError:
    HAS_VOCAB_METRICS = False    

# System metrics
cpu_usage = Gauge('system_cpu_usage_percent', 'System CPU usage percent')
ram_usage = Gauge('system_ram_usage_percent', 'System RAM usage percent')
disk_usage = Gauge('system_disk_usage_percent', 'System disk usage percent', ['device'])
net_bytes_sent = Gauge('system_net_bytes_sent', 'Network bytes sent', ['iface'])
net_bytes_recv = Gauge('system_net_bytes_recv', 'Network bytes received', ['iface'])
if HAS_GPU:
    gpu_util = Gauge('system_gpu_utilization_percent', 'GPU utilization percent', ['gpu_id'])
    gpu_mem = Gauge('system_gpu_memory_used_mb', 'GPU memory used (MB)', ['gpu_id'])
    

def collect_metrics():
    cpu_usage.set(psutil.cpu_percent())
    ram_usage.set(psutil.virtual_memory().percent)
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
            disk_usage.labels(device=part.mountpoint).set(usage.percent)
        except Exception:
            continue
    net = psutil.net_io_counters(pernic=True)
    for iface, stats in net.items():
        net_bytes_sent.labels(iface=iface).set(stats.bytes_sent)
        net_bytes_recv.labels(iface=iface).set(stats.bytes_recv)
    if HAS_GPU:
        for gpu in GPUtil.getGPUs():
            gpu_util.labels(gpu_id=gpu.id).set(gpu.load * 100)
            gpu_mem.labels(gpu_id=gpu.id).set(gpu.memoryUsed)

    # Collect vocabulary metrics
    if HAS_VOCAB_METRICS:
        try:
            collect_vocabulary_metrics()
        except Exception as e:
            print(f"Error collecting vocabulary metrics: {e}")        

def main():
    start_http_server(9000)  # Expose metrics on port 9000
    while True:
        collect_metrics()
        time.sleep(5)

if __name__ == '__main__':
    main()