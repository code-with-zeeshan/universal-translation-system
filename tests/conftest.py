"""
pytest conftest: set up project package stubs AND mock heavy deps so that
all test files can import their modules under test without real deps.
"""
import math
import os
import sys
import types
from unittest.mock import MagicMock

os.environ.setdefault("UTS_HMAC_KEY", "a" * 40)

# =========================================================================
# 1. Project package stubs (prevent __init__.py from executing)
# =========================================================================
_PROJECT_PACKAGES = [
    "utils", "data", "training", "integration",
    "evaluation", "vocabulary", "config", "connector",
    "monitoring", "tools", "encoder", 
    "encoder_core", "coordinator", "cloud_decoder",
]

for pkg in _PROJECT_PACKAGES:
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = []
        mod.__package__ = pkg
        sys.modules[pkg] = mod

# =========================================================================
# 2. Minimal working numpy (real implementations, not mocks)
# =========================================================================
_np = types.ModuleType('numpy')

def _np_mean(arr, axis=None):
    if not arr:
        return 0.0
    return sum(arr) / len(arr)

def _np_std(arr, axis=None, ddof=0):
    if not arr:
        return 0.0
    m = _np_mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))

def _np_var(arr, axis=None):
    if not arr:
        return 0.0
    m = _np_mean(arr)
    return sum((x - m) ** 2 for x in arr) / len(arr)

def _np_exp(x):
    return math.exp(x)

def _np_percentile(arr, q):
    if not arr:
        return 0.0
    s = sorted(arr)
    idx = int(len(s) * q / 100)
    return s[min(idx, len(s) - 1)]

def _np_clip(x, a, b):
    return max(a, min(b, x))

_np.mean = _np_mean
_np.std = _np_std
_np.var = _np_var
_np.exp = _np_exp
_np.percentile = _np_percentile
_np.clip = _np_clip

# =========================================================================
# 3. Helper: create recursively deep mock torch submodules
# =========================================================================

def _make_mock_submodule(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.__spec__ = None
    return mod

_TORCH_SUBMODULE_PATHS = [
    "torch.utils",
    "torch.utils.data",
    "torch.utils.checkpoint",
    "torch.distributed",
    "torch.distributed.fsdp",
    "torch.distributed.fsdp.wrap",
    "torch.optim",
    "torch.optim.lr_scheduler",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.utils",
    "torch.cuda",
    "torch.cuda.amp",
    "torch.backends",
    "torch.backends.cuda",
    "torch.backends.cudnn",
    "torch.ao",
    "torch.ao.quantization",
    "torch.quantization",
    "torch.amp",
    "torch._dynamo",
    "torch._dynamo.config",
    "torch._C",
    "torch.nested",
]

# =========================================================================
# 4. Mock torch with deep recursive hierarchy
# =========================================================================

# Tiny helper classes so isinstance checks work
class MockModule: pass
class MockLinear(MockModule): pass
class MockConv1d(MockModule): pass
class MockLSTM(MockModule): pass
class MockGRU(MockModule): pass
class MockMultiheadAttention(MockModule): pass
class MockBatchNorm1d(MockModule): pass
class MockBatchNorm2d(MockModule): pass
class MockBatchNorm3d(MockModule): pass

# Build the torch namespace as a proper module
_torch = _make_mock_submodule('torch')
_torch.__path__ = []  # Allow Python to find submodules via sys.modules

# Core types and dtypes
_torch.Module = MockModule
_torch.Tensor = MagicMock
_torch.dtype = type('dtype', (), {})
_torch.bfloat16 = _torch.dtype()
_torch.float16 = _torch.dtype()
_torch.float32 = _torch.dtype()
_torch.float64 = _torch.dtype()
_torch.qint8 = _torch.dtype()
_torch.quint8 = _torch.dtype()
_torch.long = _torch.dtype()
_torch.bool = _torch.dtype()
_torch.int64 = _torch.dtype()
_torch.int32 = _torch.dtype()

# Core functions
_torch.save = MagicMock()
_torch.load = MagicMock()
_torch.tensor = MagicMock()
_torch.cat = MagicMock()
_torch.stack = MagicMock()
_torch.norm = MagicMock()
_torch.ones_like = MagicMock()
_torch.zeros_like = MagicMock()
_torch.zeros = MagicMock()
_torch.randn = MagicMock()
_torch.randn_like = MagicMock()
_torch.full = MagicMock()
_torch.arange = MagicMock()
_torch.softmax = MagicMock()
_torch.log = MagicMock()
_torch.sqrt = MagicMock()
_torch.abs = MagicMock()
_torch.clamp = MagicMock()
_torch.mean = MagicMock()
_torch.sum = MagicMock()
_torch.max = MagicMock()
_torch.min = MagicMock()
_torch.where = MagicMock()
_torch.argmax = MagicMock()
_torch.argmin = MagicMock()
_torch.sort = MagicMock()
_torch.topk = MagicMock()
_torch.multinomial = MagicMock()
_torch.randint = MagicMock()
_torch.manual_seed = MagicMock()
_torch.seed = MagicMock()
_torch.initial_seed = MagicMock()
_torch.is_tensor = MagicMock(return_value=True)
_torch.is_floating_point = MagicMock(return_value=True)
_torch.set_default_dtype = MagicMock()
_torch.set_default_tensor_type = MagicMock()
_torch.numel = MagicMock()
_torch.no_grad = MagicMock()
_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
_torch.no_grad.return_value.__exit__ = MagicMock(return_value=None)
_torch.inference_mode = MagicMock()
_torch.inference_mode.return_value.__enter__ = MagicMock(return_value=None)
_torch.inference_mode.return_value.__exit__ = MagicMock(return_value=None)
_torch.enable_grad = MagicMock()
_torch.set_grad_enabled = MagicMock()
_torch.compile = MagicMock(return_value=MagicMock())
_torch.device = MagicMock(return_value='cpu')
_torch.channels_last = MagicMock()
_torch.contiguous_format = MagicMock()
_torch.preserve_format = MagicMock()
_torch.__version__ = '2.1.0'

# nested tensor support
_torch.nested_tensor = MagicMock()
_torch._nested_tensor_from_tensor_list = MagicMock()
_torch._nested_tensor_from_mask = MagicMock()

# nn submodule
_torch.nn = _make_mock_submodule('torch.nn')
_torch.nn.Module = MockModule
_torch.nn.Linear = MockLinear
_torch.nn.Conv1d = MockConv1d
_torch.nn.LSTM = MockLSTM
_torch.nn.GRU = MockGRU
_torch.nn.MultiheadAttention = MockMultiheadAttention
_torch.nn.BatchNorm1d = MockBatchNorm1d
_torch.nn.BatchNorm2d = MockBatchNorm2d
_torch.nn.BatchNorm3d = MockBatchNorm3d
_torch.nn.CrossEntropyLoss = MagicMock()
_torch.nn.CrossEntropyLoss.return_value = MagicMock()
_torch.nn.MSELoss = MagicMock()
_torch.nn.L1Loss = MagicMock()
_torch.nn.NLLLoss = MagicMock()
_torch.nn.BCEWithLogitsLoss = MagicMock()
_torch.nn.Embedding = MagicMock()
_torch.nn.Dropout = MagicMock()
_torch.nn.ReLU = MagicMock()
_torch.nn.GELU = MagicMock()
_torch.nn.SiLU = MagicMock()
_torch.nn.Sigmoid = MagicMock()
_torch.nn.Tanh = MagicMock()
_torch.nn.Softmax = MagicMock()
_torch.nn.LogSoftmax = MagicMock()
_torch.nn.Sequential = MagicMock()
_torch.nn.ModuleList = MagicMock()
_torch.nn.ModuleDict = MagicMock()
_torch.nn.Parameter = MagicMock()
_torch.nn.ParameterList = MagicMock()
_torch.nn.Identity = MagicMock()
_torch.nn.LayerNorm = MagicMock()
_torch.nn.RMSNorm = MagicMock()
_torch.nn.GroupNorm = MagicMock()
_torch.nn.InstanceNorm1d = MagicMock()
_torch.nn.InstanceNorm2d = MagicMock()
_torch.nn.InstanceNorm3d = MagicMock()
_torch.nn.Transformer = MagicMock()
_torch.nn.TransformerEncoder = MagicMock()
_torch.nn.TransformerDecoder = MagicMock()
_torch.nn.TransformerEncoderLayer = MagicMock()
_torch.nn.TransformerDecoderLayer = MagicMock()
_torch.nn.PReLU = MagicMock()
_torch.nn.LeakyReLU = MagicMock()
_torch.nn.ELU = MagicMock()
_torch.nn.Conv2d = MagicMock()
_torch.nn.ConvTranspose2d = MagicMock()
_torch.nn.MaxPool1d = MagicMock()
_torch.nn.MaxPool2d = MagicMock()
_torch.nn.AvgPool1d = MagicMock()
_torch.nn.AdaptiveAvgPool1d = MagicMock()
_torch.nn.AdaptiveAvgPool2d = MagicMock()
_torch.nn.Flatten = MagicMock()
_torch.nn.Unflatten = MagicMock()
_torch.nn.PixelShuffle = MagicMock()
_torch.nn.Upsample = MagicMock()
_torch.nn.ZeroPad2d = MagicMock()
_torch.nn.ConstantPad2d = MagicMock()
_torch.nn.ReflectionPad2d = MagicMock()
_torch.nn.ReplicationPad2d = MagicMock()
_torch.nn.utils = _make_mock_submodule('torch.nn.utils')
_torch.nn.utils.clip_grad_norm_ = MagicMock()
_torch.nn.utils.clip_grad_value_ = MagicMock()
_torch.nn.utils.rnn = _make_mock_submodule('torch.nn.utils.rnn')
_torch.nn.utils.rnn.pad_sequence = MagicMock()
_torch.nn.utils.rnn.pack_padded_sequence = MagicMock()
_torch.nn.utils.rnn.pad_packed_sequence = MagicMock()
_torch.nn.utils.parametrize = _make_mock_submodule('torch.nn.utils.parametrize')

# nn.functional submodule
_torch.nn.functional = _make_mock_submodule('torch.nn.functional')
_torch.nn.functional.cross_entropy = MagicMock(return_value=MagicMock())
_torch.nn.functional.cosine_similarity = MagicMock(return_value=MagicMock())
_torch.nn.functional.cosine_similarity.return_value.item = MagicMock(return_value=0.95)
_torch.nn.functional.softmax = MagicMock()
_torch.nn.functional.log_softmax = MagicMock()
_torch.nn.functional.relu = MagicMock()
_torch.nn.functional.gelu = MagicMock()
_torch.nn.functional.silu = MagicMock()
_torch.nn.functional.sigmoid = MagicMock()
_torch.nn.functional.tanh = MagicMock()
_torch.nn.functional.dropout = MagicMock()
_torch.nn.functional.embedding = MagicMock()
_torch.nn.functional.linear = MagicMock()
_torch.nn.functional.layer_norm = MagicMock()
_torch.nn.functional.normalize = MagicMock()
_torch.nn.functional.pad = MagicMock()
_torch.nn.functional.one_hot = MagicMock()
_torch.nn.functional.binary_cross_entropy = MagicMock()
_torch.nn.functional.binary_cross_entropy_with_logits = MagicMock()
_torch.nn.functional.mse_loss = MagicMock()
_torch.nn.functional.l1_loss = MagicMock()
_torch.nn.functional.nll_loss = MagicMock()
_torch.nn.functional.kl_div = MagicMock()
_torch.nn.functional.scaled_dot_product_attention = MagicMock()
_torch.nn.functional.interpolate = MagicMock()
_torch.nn.functional.avg_pool1d = MagicMock()
_torch.nn.functional.avg_pool2d = MagicMock()
_torch.nn.functional.max_pool1d = MagicMock()
_torch.nn.functional.max_pool2d = MagicMock()
_torch.nn.functional.adaptive_avg_pool1d = MagicMock()
_torch.nn.functional.adaptive_avg_pool2d = MagicMock()
_torch.nn.functional.grid_sample = MagicMock()
_torch.nn.functional.conv1d = MagicMock()
_torch.nn.functional.conv2d = MagicMock()
_torch.nn.functional.conv_transpose2d = MagicMock()
_torch.nn.functional.logsigmoid = MagicMock()

# torch.utils
_torch.utils = _make_mock_submodule('torch.utils')
_torch.utils.data = _make_mock_submodule('torch.utils.data')
_torch.utils.data.DataLoader = MagicMock()
_torch.utils.data.Dataset = MagicMock()
_torch.utils.data.IterableDataset = MagicMock()
_torch.utils.data.TensorDataset = MagicMock()
_torch.utils.data.Subset = MagicMock()
_torch.utils.data.random_split = MagicMock()
_torch.utils.data.distributed = _make_mock_submodule('torch.utils.data.distributed')
_torch.utils.data.distributed.DistributedSampler = MagicMock()

_torch.utils.checkpoint = _make_mock_submodule('torch.utils.checkpoint')
_torch.utils.checkpoint.checkpoint = MagicMock()
_torch.utils.checkpoint.checkpoint_sequential = MagicMock()

_torch.utils.data.sampler = _make_mock_submodule('torch.utils.data.sampler')
_torch.utils.data.sampler.Sampler = MagicMock()
_torch.utils.data.sampler.WeightedRandomSampler = MagicMock()
_torch.utils.data.sampler.BatchSampler = MagicMock()

_torch.utils.tensorboard = _make_mock_submodule('torch.utils.tensorboard')
_torch.utils.tensorboard.SummaryWriter = MagicMock()

_torch.utils.cpp_extension = _make_mock_submodule('torch.utils.cpp_extension')
_torch.utils.cpp_extension.CUDAExtension = MagicMock()
_torch.utils.cpp_extension.CppExtension = MagicMock()
_torch.utils.cpp_extension.load = MagicMock()

# torch.cuda
_torch.cuda = _make_mock_submodule('torch.cuda')
_torch.cuda.is_available = MagicMock(return_value=True)
_torch.cuda.device_count = MagicMock(return_value=1)
_torch.cuda.get_device_capability = MagicMock(return_value=(8, 0))
_torch.cuda.get_device_name = MagicMock(return_value="NVIDIA A100")
_torch.cuda.memory_allocated = MagicMock(return_value=2 * 1024**3)
_torch.cuda.memory_reserved = MagicMock(return_value=4 * 1024**3)
_torch.cuda.max_memory_allocated = MagicMock(return_value=3 * 1024**3)
_torch.cuda.max_memory_reserved = MagicMock(return_value=8 * 1024**3)
_torch.cuda.empty_cache = MagicMock()
_torch.cuda.reset_peak_memory_stats = MagicMock()
_torch.cuda.set_per_process_memory_fraction = MagicMock()
_torch.cuda.is_bf16_supported = MagicMock(return_value=True)
_torch.cuda.CUDAPluggableAllocator = MagicMock()
_torch.cuda.set_allocator_settings = MagicMock()
_torch.cuda.get_device_properties = MagicMock()
_torch.cuda.get_device_properties.return_value.total_memory = 80 * 1024**3
_torch.cuda.amp = _make_mock_submodule('torch.cuda.amp')
_torch.cuda.amp.GradScaler = MagicMock()
_torch.cuda.amp.autocast = MagicMock()
_torch.cuda.synchronize = MagicMock()
_torch.cuda.current_device = MagicMock(return_value=0)
_torch.cuda.device = MagicMock()
_torch.cuda.Device = MagicMock
_torch.cuda.Stream = MagicMock
_torch.cuda.set_device = MagicMock()

# torch.distributed
_torch.distributed = _make_mock_submodule('torch.distributed')
_torch.distributed.is_initialized = MagicMock(return_value=False)
_torch.distributed.get_rank = MagicMock(return_value=0)
_torch.distributed.get_world_size = MagicMock(return_value=1)
_torch.distributed.init_process_group = MagicMock()
_torch.distributed.destroy_process_group = MagicMock()
_torch.distributed.all_reduce = MagicMock()
_torch.distributed.all_gather = MagicMock()
_torch.distributed.broadcast = MagicMock()
_torch.distributed.barrier = MagicMock()
_torch.distributed.reduce = MagicMock()
_torch.distributed.gather = MagicMock()
_torch.distributed.scatter = MagicMock()
_torch.distributed.new_group = MagicMock()
_torch.distributed.GroupMember = MagicMock()
_torch.distributed.ReduceOp = MagicMock()
_torch.distributed.fsdp = _make_mock_submodule('torch.distributed.fsdp')
_torch.distributed.fsdp.FullyShardedDataParallel = MagicMock()
_torch.distributed.fsdp.FullyShardedDataParallel.return_value = MagicMock()
_torch.distributed.fsdp.wrap = _make_mock_submodule('torch.distributed.fsdp.wrap')
_torch.distributed.fsdp.wrap.transformer_auto_wrap_policy = MagicMock()
_torch.distributed.fsdp.wrap.size_based_auto_wrap_policy = MagicMock()
_torch.distributed.fsdp.wrap.always_wrap_policy = MagicMock()
_torch.distributed.fsdp.MixedPrecision = MagicMock()
_torch.distributed.fsdp.ShardingStrategy = MagicMock()
_torch.distributed.fsdp.BackwardPrefetch = MagicMock()
_torch.distributed.fsdp.CPUOffload = MagicMock()

# torch.optim
_torch.optim = _make_mock_submodule('torch.optim')
_torch.optim.AdamW = MagicMock()
_torch.optim.Adam = MagicMock()
_torch.optim.SGD = MagicMock()
_torch.optim.Adamax = MagicMock()
_torch.optim.RMSprop = MagicMock()
_torch.optim.Adagrad = MagicMock()
_torch.optim.Adadelta = MagicMock()
_torch.optim.lr_scheduler = _make_mock_submodule('torch.optim.lr_scheduler')
_torch.optim.lr_scheduler.CosineAnnealingWarmRestarts = MagicMock()
_torch.optim.lr_scheduler.CosineAnnealingLR = MagicMock()
_torch.optim.lr_scheduler.StepLR = MagicMock()
_torch.optim.lr_scheduler.MultiStepLR = MagicMock()
_torch.optim.lr_scheduler.ExponentialLR = MagicMock()
_torch.optim.lr_scheduler.ReduceLROnPlateau = MagicMock()
_torch.optim.lr_scheduler.LambdaLR = MagicMock()
_torch.optim.lr_scheduler.OneCycleLR = MagicMock()
_torch.optim.lr_scheduler.LinearLR = MagicMock()
_torch.optim.lr_scheduler.PolynomialLR = MagicMock()
_torch.optim.lr_scheduler.ConstantLR = MagicMock()
_torch.optim.lr_scheduler.SequentialLR = MagicMock()
_torch.optim.lr_scheduler.ChainedScheduler = MagicMock()
_torch.optim.Optimizer = MagicMock

# torch.backends
_torch.backends = _make_mock_submodule('torch.backends')
_torch.backends.cuda = _make_mock_submodule('torch.backends.cuda')
_torch.backends.cuda.enable_flash_sdp = MagicMock()
_torch.backends.cuda.enable_math_sdp = MagicMock()
_torch.backends.cuda.enable_mem_efficient_sdp = MagicMock()
_torch.backends.cuda.flash_sdp_enabled = MagicMock(return_value=True)
_torch.backends.cuda.math_sdp_enabled = MagicMock(return_value=False)
_torch.backends.cuda.mem_efficient_sdp_enabled = MagicMock(return_value=True)
_torch.backends.cuda.preferred_linalg_library = MagicMock()
_torch.backends.cuda.is_built = MagicMock(return_value=True)
_torch.backends.cudnn = _make_mock_submodule('torch.backends.cudnn')
_torch.backends.cudnn.allow_tf32 = MagicMock()
_torch.backends.cudnn.is_available = MagicMock(return_value=True)
_torch.backends.cudnn.version = MagicMock(return_value=8600)
_torch.backends.cudnn.enabled = True
_torch.backends.cudnn.deterministic = False
_torch.backends.cudnn.benchmark = False
_torch.backends.mps = _make_mock_submodule('torch.backends.mps')
_torch.backends.mps.is_available = MagicMock(return_value=False)
_torch.backends.mps.is_built = MagicMock(return_value=False)

# torch.ao.quantization
_torch.ao = _make_mock_submodule('torch.ao')
_torch.ao.quantization = _make_mock_submodule('torch.ao.quantization')
_torch.ao.quantization.quantize_dynamic = MagicMock(return_value=MagicMock())
_torch.ao.quantization.default_dynamic_qconfig = MagicMock()
_torch.ao.quantization.QConfig = MagicMock()
_torch.ao.quantization.MinMaxObserver = MagicMock()
_torch.ao.quantization.prepare = MagicMock(return_value=MagicMock())
_torch.ao.quantization.convert = MagicMock(return_value=MagicMock())
_torch.ao.quantization.get_default_qconfig = MagicMock()
_torch.ao.quantization.prepare_fx = MagicMock(return_value=MagicMock())
_torch.ao.quantization.convert_fx = MagicMock(return_value=MagicMock())
_torch.ao.quantization.QConfigMapping = MagicMock()
_torch.ao.quantization.QuantStub = MagicMock()
_torch.ao.quantization.DeQuantStub = MagicMock()
_torch.ao.quantization.QuantWrapper = MagicMock()
_torch.ao.quantization.observer = _make_mock_submodule('torch.ao.quantization.observer')
_torch.ao.quantization.observer.HistogramObserver = MagicMock()
_torch.ao.quantization.observer.PerChannelMinMaxObserver = MagicMock()

# torch.quantization (old API)
_torch.quantization = _make_mock_submodule('torch.quantization')
_torch.quantization.QConfig = MagicMock()
_torch.quantization.MinMaxObserver = MagicMock()
_torch.quantization.prepare = MagicMock(return_value=MagicMock())
_torch.quantization.convert = MagicMock(return_value=MagicMock())
_torch.quantization.quantize_dynamic = MagicMock()

# torch.amp
_torch.amp = _make_mock_submodule('torch.amp')
_torch.amp.GradScaler = MagicMock()
_torch.amp.autocast = MagicMock()
_torch.amp.autocast.return_value.__enter__ = MagicMock(return_value=None)
_torch.amp.autocast.return_value.__exit__ = MagicMock(return_value=None)

# torch._dynamo
_torch._dynamo = _make_mock_submodule('torch._dynamo')
_torch._dynamo.config = _make_mock_submodule('torch._dynamo.config')
_torch._dynamo.config.suppress_errors = True
_torch._dynamo.config.cache_size_limit = 512
_torch._dynamo.optimize = MagicMock()
_torch._dynamo.reset = MagicMock()

# torch._C
_torch._C = _make_mock_submodule('torch._C')
_torch._C._set_nested_tensor_enabled = MagicMock()

# torch.nested
_torch.nested = _make_mock_submodule('torch.nested')
_torch.nested.nested_tensor = MagicMock()
_torch.nested.nested_tensor_from_mask = MagicMock()
_torch.nested.nested_tensor_from_tensor_list = MagicMock()
_torch.nested.to_padded_tensor = MagicMock()

# =========================================================================
# 5. Mock other heavy third-party packages
# =========================================================================

# --- wandb ---
_wandb = MagicMock()
_wandb.init = MagicMock()
_wandb.log = MagicMock()
_wandb.config = MagicMock()
_wandb.run = MagicMock()
_wandb.finish = MagicMock()

# --- psutil ---
_psutil = MagicMock()
_psutil.cpu_percent = MagicMock(return_value=45.0)
_psutil.virtual_memory = MagicMock()
_psutil.virtual_memory.return_value.percent = 60.0
_psutil.virtual_memory.return_value.available = 8 * 1024**3
_psutil.Process = MagicMock()
_psutil.Process.return_value.memory_info = MagicMock()
_psutil.Process.return_value.memory_info.return_value.rss = 500 * 1024**2
_psutil.Process.return_value.memory_info.return_value.vms = 1 * 1024**3
_psutil.Process.return_value.memory_percent = MagicMock(return_value=10.0)

# --- prometheus_client ---
_prometheus = MagicMock()
_prometheus.Gauge = MagicMock()
_prometheus.Counter = MagicMock()
_prometheus.Histogram = MagicMock()
_prometheus.Summary = MagicMock()
_prometheus.Info = MagicMock()
_prometheus.Enum = MagicMock()
_prometheus.start_http_server = MagicMock()
_prometheus.REGISTRY = MagicMock()

# --- safetensors ---
_safetensors = types.ModuleType('safetensors')
_safe_torch = types.ModuleType('safetensors.torch')
_safe_torch.save_file = MagicMock()
_safe_torch.load_file = MagicMock()
_safetensors.torch = _safe_torch

# --- yaml ---
class _YAMLError(Exception):
    pass

_yaml = MagicMock()
_yaml.YAMLError = _YAMLError
_yaml.safe_load = MagicMock()
_yaml.dump = MagicMock()
_yaml.safe_dump = MagicMock()
_yaml.load = MagicMock()
_yaml.add_constructor = MagicMock()
_yaml.add_representer = MagicMock()

# --- sacrebleu ---
_sacrebleu = MagicMock()
_sacrebleu.corpus_bleu = MagicMock(return_value=MagicMock())
_sacrebleu.corpus_bleu.return_value.score = 25.0

# --- nvidia_ml_py3 ---
_nvml = MagicMock()
_nvml.nvmlInit = MagicMock()
_nvml.nvmlDeviceGetHandleByIndex = MagicMock()
_nvml.nvmlDeviceGetUtilizationRates = MagicMock()
_nvml.nvmlDeviceGetUtilizationRates.return_value.gpu = 80
_nvml.nvmlDeviceGetMemoryInfo = MagicMock()
_nvml.nvmlDeviceGetMemoryInfo.return_value.used = 40 * 1024**3
_nvml.nvmlDeviceGetMemoryInfo.return_value.total = 80 * 1024**3

# --- pydantic ---
_pydantic = MagicMock()
_pydantic.BaseModel = type('BaseModel', (), {'__init__': MagicMock(return_value=None)})
_pydantic.Field = MagicMock()
_pydantic.validator = MagicMock()
_pydantic.ValidationError = type('ValidationError', (Exception,), {})

# --- jwt (PyJWT) ---
_jwt = types.ModuleType('jwt')
_jwt.encode = MagicMock(return_value=b'token')
_jwt.decode = MagicMock(return_value={'sub': 'test'})
_jwt.PyJWTError = type('PyJWTError', (Exception,), {})
_jwt.ExpiredSignatureError = type('ExpiredSignatureError', (_jwt.PyJWTError,), {})
_jwt.InvalidTokenError = type('InvalidTokenError', (_jwt.PyJWTError,), {})
_jwt.InvalidSignatureError = type('InvalidSignatureError', (_jwt.PyJWTError,), {})
_jwt.DecodeError = type('DecodeError', (_jwt.PyJWTError,), {})
_jwt.InvalidAudienceError = type('InvalidAudienceError', (_jwt.PyJWTError,), {})
_jwt.InvalidIssuerError = type('InvalidIssuerError', (_jwt.PyJWTError,), {})
_jwt.InvalidIssuedAtError = type('InvalidIssuedAtError', (_jwt.PyJWTError,), {})
_jwt.ImmatureSignatureError = type('ImmatureSignatureError', (_jwt.PyJWTError,), {})
_jwt.MissingRequiredClaimError = type('MissingRequiredClaimError', (_jwt.PyJWTError,), {})

# --- cryptography ---
_crypto = types.ModuleType('cryptography')
_crypto_hazmat = types.ModuleType('cryptography.hazmat')
_crypto_primitives = types.ModuleType('cryptography.hazmat.primitives')
_crypto_serialization = types.ModuleType('cryptography.hazmat.primitives.serialization')
_crypto_hashes = types.ModuleType('cryptography.hazmat.primitives.hashes')
_crypto_asymmetric = types.ModuleType('cryptography.hazmat.primitives.asymmetric')
_crypto_rsa = types.ModuleType('cryptography.hazmat.primitives.asymmetric.rsa')
_crypto_backends = types.ModuleType('cryptography.hazmat.backends')
_crypto_default_backend = types.ModuleType('cryptography.hazmat.backends.default_backend')
_crypto_fernet = types.ModuleType('cryptography.fernet')
_crypto_kdf = types.ModuleType('cryptography.hazmat.primitives.kdf')
_crypto_pbkdf2 = types.ModuleType('cryptography.hazmat.primitives.kdf.pbkdf2')

_crypto_serialization.Encoding = MagicMock()
_crypto_serialization.PublicFormat = MagicMock()
_crypto_serialization.PrivateFormat = MagicMock()
_crypto_serialization.NoEncryption = MagicMock()
_crypto_serialization.BestAvailableEncryption = MagicMock()
_crypto_serialization.load_pem_public_key = MagicMock()
_crypto_serialization.load_pem_private_key = MagicMock()
_crypto_serialization.load_ssh_public_key = MagicMock()

_crypto_hashes.Hash = MagicMock()
_crypto_hashes.SHA256 = MagicMock()

class MockRSAPrivateKey:
    pass
class MockRSAPublicKey:
    def public_bytes(self, *args, **kwargs):
        return b''
    def public_numbers(self):
        return MagicMock()

_crypto_rsa.RSAPrivateKey = MockRSAPrivateKey
_crypto_rsa.RSAPublicKey = MockRSAPublicKey
_crypto_rsa.generate_private_key = MagicMock(return_value=MockRSAPrivateKey())

_crypto_default_backend.default_backend = MagicMock()

_crypto_fernet.Fernet = MagicMock()

_crypto_pbkdf2.PBKDF2HMAC = MagicMock()

_crypto.hazmat = _crypto_hazmat
_crypto_hazmat.primitives = _crypto_primitives
_crypto_primitives.serialization = _crypto_serialization
_crypto_primitives.hashes = _crypto_hashes
_crypto_primitives.asymmetric = _crypto_asymmetric
_crypto_asymmetric.rsa = _crypto_rsa
_crypto_primitives.kdf = _crypto_kdf
_crypto_kdf.pbkdf2 = _crypto_pbkdf2
_crypto_hazmat.backends = _crypto_backends
_crypto_backends.default_backend = _crypto_default_backend

# --- keyring ---
_keyring = types.ModuleType('keyring')
_keyring.get_password = MagicMock(return_value=None)
_keyring.set_password = MagicMock()
_keyring.delete_password = MagicMock()

# --- msgpack ---
_msgpack = types.ModuleType('msgpack')
_msgpack.packb = MagicMock(return_value=b'')
_msgpack.unpackb = MagicMock(return_value={})
_msgpack.PackException = type('PackException', (Exception,), {})
_msgpack.UnpackException = type('UnpackException', (Exception,), {})

# --- fastapi ---
_fastapi = types.ModuleType('fastapi')
_fastapi.FastAPI = MagicMock()
_fastapi.Request = MagicMock()
_fastapi.Header = MagicMock()
_fastapi.Depends = MagicMock()
_fastapi.HTTPException = MagicMock()
_fastapi.APIRouter = MagicMock()
_fastapi.Query = MagicMock()
_fastapi.Path = MagicMock()
_fastapi.Body = MagicMock()
_fastapi.Form = MagicMock()
_fastapi.File = MagicMock()
_fastapi.UploadFile = MagicMock()
_fastapi.Response = MagicMock()
_fastapi.responses = types.ModuleType('fastapi.responses')
_fastapi.responses.JSONResponse = MagicMock()
_fastapi.responses.PlainTextResponse = MagicMock()
_fastapi.responses.HTMLResponse = MagicMock()
_fastapi.responses.RedirectResponse = MagicMock()
_fastapi.security = types.ModuleType('fastapi.security')
_fastapi.security.HTTPBearer = MagicMock()
_fastapi.security.HTTPAuthorizationCredentials = MagicMock()
_fastapi.testclient = types.ModuleType('fastapi.testclient')
_fastapi.testclient.TestClient = MagicMock()

# --- starlette ---
_starlette = types.ModuleType('starlette')
_starlette.testclient = types.ModuleType('starlette.testclient')
_starlette.testclient.TestClient = MagicMock()

# --- litserve ---
_litserve = types.ModuleType('litserve')
_litserve.LitServer = MagicMock()
_litserve.LitAPI = MagicMock()

# --- opentelemetry ---
_otel = types.ModuleType('opentelemetry')
_otel_trace = types.ModuleType('opentelemetry.trace')
_otel_trace.get_tracer = MagicMock(return_value=MagicMock())
_otel_trace.get_tracer.return_value.start_span = MagicMock()
_otel_trace.get_tracer.return_value.start_span.return_value.__enter__ = MagicMock()
_otel_trace.get_tracer.return_value.start_span.return_value.__exit__ = MagicMock()
_otel.trace = _otel_trace

# --- urllib3 ---
_urllib3 = MagicMock()
_urllib3.disable_warnings = MagicMock()

# --- requests ---
_requests = MagicMock()
_requests.get = MagicMock()
_requests.post = MagicMock()
_requests.put = MagicMock()
_requests.delete = MagicMock()
_requests.head = MagicMock()
_requests.options = MagicMock()
_requests.patch = MagicMock()
_requests.exceptions = MagicMock()
_requests.exceptions.RequestException = type('RequestException', (Exception,), {})
_requests.exceptions.ConnectionError = type('ConnectionError', (_requests.exceptions.RequestException,), {})
_requests.exceptions.Timeout = type('Timeout', (_requests.exceptions.RequestException,), {})
_requests.exceptions.HTTPError = type('HTTPError', (_requests.exceptions.RequestException,), {})
_requests.codes = MagicMock()

# --- tracemalloc --- (stdlib, but resource_tracker imports it)
import tracemalloc as _real_tracemalloc
_tracemalloc = _real_tracemalloc

# =========================================================================
# 6. Install all mocks into sys.modules
# =========================================================================
_HEAVY_MOCKS = {
    'numpy': _np,
    'torch': _torch,
    'torch.nn': _torch.nn,
    'torch.nn.functional': _torch.nn.functional,
    'torch.nn.utils': _torch.nn.utils,
    'torch.nn.utils.rnn': _torch.nn.utils.rnn,
    'torch.utils': _torch.utils,
    'torch.utils.data': _torch.utils.data,
    'torch.utils.data.distributed': _torch.utils.data.distributed,
    'torch.utils.data.sampler': _torch.utils.data.sampler,
    'torch.utils.checkpoint': _torch.utils.checkpoint,
    'torch.utils.tensorboard': _torch.utils.tensorboard,
    'torch.utils.cpp_extension': _torch.utils.cpp_extension,
    'torch.cuda': _torch.cuda,
    'torch.cuda.amp': _torch.cuda.amp,
    'torch.distributed': _torch.distributed,
    'torch.distributed.fsdp': _torch.distributed.fsdp,
    'torch.distributed.fsdp.wrap': _torch.distributed.fsdp.wrap,
    'torch.optim': _torch.optim,
    'torch.optim.lr_scheduler': _torch.optim.lr_scheduler,
    'torch.ao': _torch.ao,
    'torch.ao.quantization': _torch.ao.quantization,
    'torch.ao.quantization.observer': _torch.ao.quantization.observer,
    'torch.quantization': _torch.quantization,
    'torch.amp': _torch.amp,
    'torch.backends': _torch.backends,
    'torch.backends.cuda': _torch.backends.cuda,
    'torch.backends.cudnn': _torch.backends.cudnn,
    'torch.backends.mps': _torch.backends.mps,
    'torch._dynamo': _torch._dynamo,
    'torch._dynamo.config': _torch._dynamo.config,
    'torch._C': _torch._C,
    'torch.nested': _torch.nested,
    'wandb': _wandb,
    'psutil': _psutil,
    'prometheus_client': _prometheus,
    'safetensors': _safetensors,
    'safetensors.torch': _safe_torch,
    'yaml': _yaml,
    'sacrebleu': _sacrebleu,
    'nvidia_ml_py3': _nvml,
    'pydantic': _pydantic,
    'jwt': _jwt,
    'cryptography': _crypto,
    'cryptography.hazmat': _crypto_hazmat,
    'cryptography.hazmat.primitives': _crypto_primitives,
    'cryptography.hazmat.primitives.serialization': _crypto_serialization,
    'cryptography.hazmat.primitives.hashes': _crypto_hashes,
    'cryptography.hazmat.primitives.asymmetric': _crypto_asymmetric,
    'cryptography.hazmat.primitives.asymmetric.rsa': _crypto_rsa,
    'cryptography.hazmat.backends': _crypto_backends,
    'cryptography.hazmat.backends.default_backend': _crypto_default_backend,
    'cryptography.fernet': _crypto_fernet,
    'cryptography.hazmat.primitives.kdf': _crypto_kdf,
    'cryptography.hazmat.primitives.kdf.pbkdf2': _crypto_pbkdf2,
    'keyring': _keyring,
    'msgpack': _msgpack,
    'fastapi': _fastapi,
    'fastapi.responses': _fastapi.responses,
    'fastapi.security': _fastapi.security,
    'fastapi.testclient': _fastapi.testclient,
    'starlette': _starlette,
    'starlette.testclient': _starlette.testclient,
    'litserve': _litserve,
    'opentelemetry': _otel,
    'opentelemetry.trace': _otel_trace,
    'urllib3': _urllib3,
    'requests': _requests,
    'requests.exceptions': _requests.exceptions,
    'tracemalloc': _tracemalloc,
}

for mod_name, mod_mock in _HEAVY_MOCKS.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mod_mock

# =========================================================================
# 7. Pre-mock project submodules that have heavy/chain imports
# =========================================================================

def _make_mock_module(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod

_PROJECT_SUBMODULE_MOCKS = {
    'config.schemas': _make_mock_module('schemas', {
        'RootConfig': MagicMock(),
        'DataConfig': MagicMock(),
        'ModelConfig': MagicMock(),
        'TrainingConfig': MagicMock(),
        'MemoryConfig': MagicMock(),
        'VocabularyConfig': MagicMock(),
        'EncoderConfig': MagicMock(),
        'DecoderConfig': MagicMock(),
        'CoordinatorConfig': MagicMock(),
        'CircuitBreakerConfig': MagicMock(),
        'MonitoringConfig': MagicMock(),
        'SystemConfig': MagicMock(),
        'load_config': MagicMock(),
        'load_system_config': MagicMock(),
        'load_pydantic_config': MagicMock(),
    }),
    'config.config_models': _make_mock_module('config_models', {
        'EncoderConfig': MagicMock(), 'DecoderConfig': MagicMock(),
        'CoordinatorConfig': MagicMock(), 'CircuitBreakerConfig': MagicMock(),
        'MonitoringConfig': MagicMock(), 'TrainingConfig': MagicMock(),
        'SystemConfig': MagicMock(), 'load_config': MagicMock(),
    }),
    'utils.exceptions': _make_mock_module('exceptions', {
        'UniversalTranslationError': type('UniversalTranslationError', (Exception,), {}),
        'DataError': type('DataError', (Exception,), {}),
        'VocabularyError': type('VocabularyError', (Exception,), {}),
        'ModelError': type('ModelError', (Exception,), {}),
        'ConfigurationError': type('ConfigurationError', (Exception,), {}),
        'TrainingError': type('TrainingError', (Exception,), {}),
        'InferenceError': type('InferenceError', (Exception,), {}),
        'ResourceError': type('ResourceError', (Exception,), {}),
        'SecurityError': type('SecurityError', (Exception,), {}),
        'LoggingError': type('LoggingError', (Exception,), {}),
        'NetworkError': type('NetworkError', (Exception,), {}),
        'TimeoutError': type('TimeoutError', (Exception,), {}),
        'AuthenticationError': type('AuthenticationError', (Exception,), {}),
        'AuthorizationError': type('AuthorizationError', (Exception,), {}),
        'MemoryError': type('MemoryError', (Exception,), {}),
        'ThreadingError': type('ThreadingError', (Exception,), {}),
        'ValidationError': type('ValidationError', (Exception,), {}),
    }),
    'utils.constants': _make_mock_module('constants', {
        'CHECKPOINT_DIR': 'checkpoints',
        'MODELS_PRODUCTION_DIR': 'models/production',
        'ENCODER_MODEL_FILENAME': 'encoder.pt',
        'DECODER_MODEL_FILENAME': 'decoder.pt',
        'VOCAB_DIR': 'vocabulary/vocab',
        'LOG_DIR': 'logs',
        'TRAINING_REPORT_FILENAME': 'training_report.json',
        'CONFIG_DIR': 'config',
        'BASE_CONFIG_FILENAME': 'base.yaml',
        'DEFAULT_BATCH_SIZE': 64,
        'DEFAULT_TIMEOUT': 30,
        'VOCAB_SIZE': 32000,
        'VOCAB_SPECIAL_TOKENS': ['<pad>', '<unk>', '<bos>', '<eos>'],
        'VOCAB_PAD_ID': 0,
        'VOCAB_UNK_ID': 1,
        'VOCAB_BOS_ID': 2,
        'VOCAB_EOS_ID': 3,
        'MAX_CACHE_SIZE': 10000,
        'MAX_MEMORY_USAGE': 1024 * 1024 * 1024,
        'TOKEN_EXPIRATION': 1800,
        'API_VERSION': '1.0.0',
        'SUPPORTED_VOCAB_FORMAT': '1',
        'MODELS_DIR': 'models',
        'DATA_PROCESSED_DIR': 'data/processed',
        'BENCHMARK_RESULTS_FILENAME': 'benchmark_results.json',
    }),
    'utils.credential_manager': _make_mock_module('credential_manager', {
        'CredentialManager': MagicMock(),
        'credential_manager': MagicMock(),
        'get_credential': MagicMock(return_value=None),
        'set_credential': MagicMock(),
        'delete_credential': MagicMock(return_value=True),
    }),
    'utils.thread_safety': _make_mock_module('thread_safety', {
        'document_thread_safety': MagicMock(side_effect=lambda cls, level, desc='': cls),
        'document_method_thread_safety': MagicMock(side_effect=lambda method, level, desc='': method),
        'thread_safe': MagicMock(side_effect=lambda f: f),
        'get_thread_safety_info': MagicMock(return_value={}),
        'generate_thread_safety_report': MagicMock(return_value={}),
        'THREAD_SAFETY_NONE': 'none',
        'THREAD_SAFETY_EXTERNAL': 'external',
        'THREAD_SAFETY_INTERNAL': 'internal',
        'THREAD_SAFETY_IMMUTABLE': 'immutable',
        '_thread_safety_registry': {},
        '_registry_lock': MagicMock(),
    }),
    'utils.secure_serialization': _make_mock_module('secure_serialization', {
        'secure_serialize_json': MagicMock(return_value=''),
        'secure_deserialize_json': MagicMock(return_value={}),
        'safe_deserialize_json': MagicMock(return_value={}),
        'secure_serialize_msgpack': MagicMock(return_value=b''),
        'secure_deserialize_msgpack': MagicMock(return_value={}),
        'safe_deserialize_msgpack': MagicMock(return_value={}),
        'safe_deserialize_pickle': MagicMock(return_value={}),
        'secure_serialize_json_compressed': MagicMock(return_value=''),
        'secure_deserialize_json_compressed': MagicMock(return_value={}),
        'secure_serialize_with_version': MagicMock(return_value=''),
        'secure_deserialize_with_version': MagicMock(return_value={}),
        'secure_deserialize_with_schema': MagicMock(return_value={}),
        'validate_type': MagicMock(),
        'SecurityError': type('SecurityError', (Exception,), {}),
    }),
    'utils.secrets_bootstrap': _make_mock_module('secrets_bootstrap', {
        'bootstrap_secrets': MagicMock(),
        'get_secret': MagicMock(return_value=None),
        'is_strong_secret': MagicMock(return_value=True),
        'validate_runtime_secrets': MagicMock(),
        'rotate_secret_if_expired': MagicMock(return_value=(False, None)),
        'is_secret_expired': MagicMock(return_value=False),
    }),
    'utils.jwks_utils': _make_mock_module('jwks_utils', {
        'build_jwks_from_env': MagicMock(return_value=[]),
        'diff_kids': MagicMock(return_value=([], [])),
    }),
    'utils.jwt_auth': _make_mock_module('jwt_auth', {}),
    'utils.common_utils': _make_mock_module('common_utils', {
        'DirectoryManager': MagicMock(),
        'ImportCleaner': MagicMock(),
    }),
    'utils.resource_tracker': _make_mock_module('resource_tracker', {
        'ResourceTracker': MagicMock(),
        'resource_tracker': MagicMock(),
        'ResourceTracked': MagicMock(),
        'track_resources': MagicMock(side_effect=lambda f: f),
    }),
    'utils.validation_decorators': _make_mock_module('validation_decorators', {}),
    'utils.base_classes': _make_mock_module('base_classes', {}),
    'utils.dataset_classes': _make_mock_module('dataset_classes', {'ModernParallelDataset': MagicMock()}),
    'utils.unified_validation': _make_mock_module('unified_validation', {'InputValidator': MagicMock()}),
    'utils.logging_config': _make_mock_module('logging_config', {'setup_logging': MagicMock()}),
    'utils.gpu_utils': _make_mock_module('gpu_utils', {
        'optimize_gpu_memory': MagicMock(),
        'get_gpu_memory_info': MagicMock(return_value={}),
    }),
    'monitoring.health_service': _make_mock_module('health_service', {'start_health_service': MagicMock()}),
    'monitoring.metrics': _make_mock_module('metrics', {
        'JWKS_RELOADS_SUCCESS': None,
        'JWKS_RELOADS_FAILURE': None,
        'JWKS_KEYS': None,
    }),
    'runtime.vocabulary.manager': _make_mock_module('manager', {
        'UnifiedVocabularyManager': MagicMock(),
        'VocabularyMode': MagicMock(),
    }),
    'integration.system_config': _make_mock_module('system_config', {
        'IntegrationSystemConfig': MagicMock(),
        'SystemConfig': MagicMock(),
    }),
    'integration.system': _make_mock_module('system', {'UniversalTranslationSystem': MagicMock()}),
    'integration.connect_all_systems': _make_mock_module('connect_all_systems', {
        'integrate_full_pipeline': MagicMock(),
    }),
    'pipeline.training.trainer': _make_mock_module('trainer', {'train_intelligent': MagicMock()}),
    'pipeline.training.strategy': _make_mock_module('strategy', {
        'TrainingStrategy': MagicMock(),
    }),
    'pipeline.training.hardware': _make_mock_module('hardware', {
        'HardwareProfile': MagicMock(),
        'find_free_port': MagicMock(return_value=12345),
        'launch_distributed_intelligent_training': MagicMock(),
    }),
    'pipeline.training.memory.config': _make_mock_module('config', {
        'MemoryConfig': MagicMock(),
    }),
    'pipeline.training.memory.trainer': _make_mock_module('trainer', {
        'MemoryConfig': MagicMock(),
        'MemoryOptimizedTrainer': MagicMock(),
        'MemoryTracker': MagicMock(),
        'DynamicBatchSizer': MagicMock(),
        'create_modern_training_setup': MagicMock(return_value={}),
        'benchmark_training_speed': MagicMock(return_value={}),
    }),
    'pipeline.training.memory.batch_sizer': _make_mock_module('batch_sizer', {
        'DynamicBatchSizer': MagicMock(),
    }),
    'pipeline.training.utils': _make_mock_module('utils', {
        'BaseTrainer': MagicMock(),
        'check_convergence': MagicMock(return_value=False),
        'find_convergence_step': MagicMock(return_value=None),
        'create_training_report': MagicMock(return_value={}),
        'calculate_gradient_norm': MagicMock(return_value=0.0),
        'create_optimizer_with_param_groups': MagicMock(return_value=MagicMock()),
        'get_adaptive_gradient_clipping_value': MagicMock(return_value=1.0),
        'get_learning_rate_schedule': MagicMock(return_value=[]),
        'get_training_diagnostics': MagicMock(return_value={}),
        'save_training_state': MagicMock(),
    }),
    'utils.artifact_store': _make_mock_module('artifact_store', {
        'ArtifactStore': MagicMock(),
        'StoreConfig': MagicMock(),
    }),
    'utils.rate_limiter': _make_mock_module('rate_limiter', {
        'RateLimiter': MagicMock(),
    }),
    'utils.logging_config': _make_mock_module('logging_config', {
        'setup_logging': MagicMock(),
        'LoggingSensitiveDataFilter': MagicMock(),
    }),
    'pipeline.training.samplers': _make_mock_module('samplers', {
        'TemperatureSampler': MagicMock(),
    }),
    'pipeline.data.state': _make_mock_module('state', {
        'PipelineStage': MagicMock(),
        'PipelineState': MagicMock(),
    }),
    'evaluation.evaluator': _make_mock_module('evaluator', {
        'TranslationPair': MagicMock(),
    }),
    'pipeline.vocabulary.config': _make_mock_module('config', {
        'CreationMode': MagicMock(),
        'UnifiedVocabConfig': MagicMock(),
        'VocabStats': MagicMock(),
        'LanguageGroup': MagicMock(),
    }),
    'encoder.universal_encoder': _make_mock_module('universal_encoder', {
        'UniversalEncoder': MagicMock(),
    }),
    'encoder.custom_layers': _make_mock_module('custom_layers', {
        'RotaryEmbedding': MagicMock(),
        'CustomTransformerEncoderLayer': MagicMock(),
    }),
    'pipeline.training.quantization.encoder': _make_mock_module('encoder', {
        'EncoderQuantizer': MagicMock(),
    }),
    'pipeline.training.quantization.pipeline': _make_mock_module('pipeline', {
        'QualityPreservingQuantizer': MagicMock(),
    }),
    'pipeline.training.quantization.quality': _make_mock_module('quality', {
        'QualityComparator': MagicMock(),
    }),
    'pipeline.training.analytics': _make_mock_module('analytics', {
        'TrainingAnalytics': MagicMock(),
    }),
    'integration.system_health': _make_mock_module('system_health', {
        'SystemHealthMonitor': MagicMock(),
    }),
    'integration.translation_api': _make_mock_module('translation_api', {
        'translate': MagicMock(),
        'translate_async': MagicMock(),
        'translate_batch_async': MagicMock(),
        'evaluate': MagicMock(),
        'evaluate_async': MagicMock(),
        'integrate_full_pipeline': MagicMock(),
        'integrate_full_pipeline_async': MagicMock(),
    }),
    'encoding.base64': _make_mock_module('base64', {}),
    'cloud_decoder.optimized_decoder': _make_mock_module('optimized_decoder', {
        'OptimizedUniversalDecoder': MagicMock(),
        'OptimizedDecoderLayer': MagicMock(),
        'ContinuousBatcher': MagicMock(),
        'startup_validation': MagicMock(),
        'app': MagicMock(),
    }),
    'coordinator.advanced_coordinator': _make_mock_module('advanced_coordinator', {
        'DASHBOARD_TEMPLATE': MagicMock(),
        'DecoderNodeSchema': MagicMock(),
        'app': MagicMock(),
    }),
}

for mod_name, mod in _PROJECT_SUBMODULE_MOCKS.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mod
