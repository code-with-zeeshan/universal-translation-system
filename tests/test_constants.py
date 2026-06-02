"""
Tests for utils.constants - centralized constants with environment overrides.
"""

import os
import json
import importlib
import pytest
import utils.constants


class TestDefaultValues:
    """Verify all constants have their expected default values."""

    def test_string_constants(self):
        assert utils.constants.DEFAULT_ENCODING == "utf-8"
        assert utils.constants.API_VERSION == "1.0.0"
        assert utils.constants.SUPPORTED_VOCAB_FORMAT == "1"

    def test_path_constants(self):
        assert utils.constants.DEFAULT_CONFIG_PATH == "config/default_config.json"
        assert utils.constants.DEFAULT_MODEL_PATH == "models/default_model"
        assert utils.constants.DEFAULT_VOCAB_PATH == "vocabulary/default_vocab"
        assert utils.constants.DEFAULT_LOG_PATH == "logs/system.log"
        assert utils.constants.VOCAB_DIR == "vocabulary/vocab"
        assert utils.constants.CONFIG_DIR == "config"
        assert utils.constants.CONFIGS_DIR == "config"
        assert utils.constants.LOG_DIR == "logs"
        assert utils.constants.DATA_RAW_DIR == "data/raw"
        assert utils.constants.DATA_PROCESSED_DIR == "data/processed"
        assert utils.constants.DATA_CACHE_DIR == "data/cache"

    def test_model_paths(self):
        assert utils.constants.MODELS_DIR == "models"
        assert utils.constants.MODELS_PRODUCTION_DIR == "models/production"
        assert utils.constants.MODELS_ENCODER_DIR == "models/encoder"
        assert utils.constants.MODELS_DECODER_DIR == "models/decoder"
        assert utils.constants.CHECKPOINT_DIR == "checkpoints"

    def test_numeric_constants(self):
        assert utils.constants.DEFAULT_TIMEOUT == 30
        assert utils.constants.MAX_RETRY_COUNT == 3
        assert utils.constants.DEFAULT_BATCH_SIZE == 64
        assert utils.constants.DEFAULT_BUFFER_SIZE == 8192
        assert utils.constants.MAX_CACHE_SIZE == 10000
        assert utils.constants.DEFAULT_CACHE_TTL == 3600
        assert utils.constants.TOKEN_EXPIRATION == 1800
        assert utils.constants.REFRESH_TOKEN_EXPIRATION == 604800
        assert utils.constants.PASSWORD_MIN_LENGTH == 8
        assert utils.constants.PASSWORD_MAX_LENGTH == 128
        assert utils.constants.MAX_LOGIN_ATTEMPTS == 5
        assert utils.constants.LOCKOUT_DURATION == 900
        assert utils.constants.API_RATE_LIMIT == 100
        assert utils.constants.API_BURST_LIMIT == 20
        assert utils.constants.API_TIMEOUT == 30

    def test_encoder_constants(self):
        assert utils.constants.ENCODER_EMBEDDING_DIM == 512
        assert utils.constants.ENCODER_HIDDEN_DIM == 1024
        assert utils.constants.ENCODER_NUM_LAYERS == 6
        assert utils.constants.ENCODER_NUM_HEADS == 8
        assert utils.constants.ENCODER_DROPOUT == 0.1
        assert utils.constants.ENCODER_MAX_LENGTH == 512

    def test_decoder_constants(self):
        assert utils.constants.DECODER_EMBEDDING_DIM == 512
        assert utils.constants.DECODER_HIDDEN_DIM == 1024
        assert utils.constants.DECODER_NUM_LAYERS == 6
        assert utils.constants.DECODER_NUM_HEADS == 8
        assert utils.constants.DECODER_DROPOUT == 0.1
        assert utils.constants.DECODER_MAX_LENGTH == 512

    def test_vocab_constants(self):
        assert utils.constants.VOCAB_SIZE == 32000
        assert utils.constants.VOCAB_MIN_FREQUENCY == 5
        assert utils.constants.VOCAB_SPECIAL_TOKENS == ["<pad>", "<unk>", "<bos>", "<eos>"]
        assert utils.constants.VOCAB_PAD_ID == 0
        assert utils.constants.VOCAB_UNK_ID == 1
        assert utils.constants.VOCAB_BOS_ID == 2
        assert utils.constants.VOCAB_EOS_ID == 3

    def test_http_status_codes(self):
        assert utils.constants.HTTP_OK == 200
        assert utils.constants.HTTP_CREATED == 201
        assert utils.constants.HTTP_ACCEPTED == 202
        assert utils.constants.HTTP_BAD_REQUEST == 400
        assert utils.constants.HTTP_UNAUTHORIZED == 401
        assert utils.constants.HTTP_FORBIDDEN == 403
        assert utils.constants.HTTP_NOT_FOUND == 404
        assert utils.constants.HTTP_SERVER_ERROR == 500

    def test_error_messages(self):
        assert utils.constants.ERROR_INVALID_INPUT == "Invalid input provided"
        assert utils.constants.ERROR_UNAUTHORIZED == "Unauthorized access"
        assert utils.constants.ERROR_RESOURCE_NOT_FOUND == "Resource not found"
        assert utils.constants.ERROR_INTERNAL_SERVER == "Internal server error"
        assert utils.constants.ERROR_RATE_LIMITED == "Rate limit exceeded"
        assert utils.constants.ERROR_INVALID_TOKEN == "Invalid or expired token"

    def test_memory_sizes(self):
        assert utils.constants.MAX_MEMORY_USAGE == 1073741824
        assert utils.constants.MAX_FILE_SIZE == 104857600

    def test_filenames(self):
        assert utils.constants.ENCODER_MODEL_FILENAME == "encoder.pt"
        assert utils.constants.DECODER_MODEL_FILENAME == "decoder.pt"
        assert utils.constants.BEST_MODEL_FILENAME == "best_model.pt"
        assert utils.constants.BASE_CONFIG_FILENAME == "base.yaml"
        assert utils.constants.VERSION_CONFIG_FILENAME == "version-config.json"
        assert utils.constants.BENCHMARK_RESULTS_FILENAME == "benchmark_results.json"
        assert utils.constants.TRAINING_REPORT_FILENAME == "training_report.json"
        assert utils.constants.EMERGENCY_CHECKPOINT_FILENAME == "emergency_checkpoint.pt"
        assert utils.constants.QUANTIZATION_REPORT_FILENAME == "quantization_report.json"


class TestEnvOverrides:
    """Test that environment variables override defaults correctly."""

    @pytest.mark.parametrize("key,value,attr,expected", [
        ("DEFAULT_TIMEOUT", "45", "DEFAULT_TIMEOUT", 45),
        ("MAX_RETRY_COUNT", "10", "MAX_RETRY_COUNT", 10),
        ("DEFAULT_BATCH_SIZE", "128", "DEFAULT_BATCH_SIZE", 128),
        ("TOKEN_EXPIRATION", "3600", "TOKEN_EXPIRATION", 3600),
    ])
    def test_env_override_int(self, monkeypatch, key, value, attr, expected):
        monkeypatch.setenv(key, value)
        importlib.reload(utils.constants)
        assert getattr(utils.constants, attr) == expected
        importlib.reload(utils.constants)

    @pytest.mark.parametrize("key,value,attr,expected", [
        ("ENCODER_DROPOUT", "0.25", "ENCODER_DROPOUT", 0.25),
        ("DECODER_DROPOUT", "0.5", "DECODER_DROPOUT", 0.5),
    ])
    def test_env_override_float(self, monkeypatch, key, value, attr, expected):
        monkeypatch.setenv(key, value)
        importlib.reload(utils.constants)
        assert getattr(utils.constants, attr) == expected
        importlib.reload(utils.constants)

    def test_env_override_str(self, monkeypatch):
        monkeypatch.setenv("API_VERSION", "2.0.0")
        importlib.reload(utils.constants)
        assert utils.constants.API_VERSION == "2.0.0"
        importlib.reload(utils.constants)

    @pytest.mark.parametrize("key,value,attr,expected", [
        ("VOCAB_SPECIAL_TOKENS", '["<pad>","<unk>","<s>","</s>"]', "VOCAB_SPECIAL_TOKENS", ["<pad>", "<unk>", "<s>", "</s>"]),
    ])
    def test_env_override_list_json(self, monkeypatch, key, value, attr, expected):
        monkeypatch.setenv(key, value)
        importlib.reload(utils.constants)
        assert getattr(utils.constants, attr) == expected
        importlib.reload(utils.constants)

    def test_env_override_list_comma(self, monkeypatch):
        monkeypatch.setenv("VOCAB_SPECIAL_TOKENS", "a, b, c")
        importlib.reload(utils.constants)
        assert utils.constants.VOCAB_SPECIAL_TOKENS == ["a", "b", "c"]
        importlib.reload(utils.constants)

    def test_uts_prefix_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_TIMEOUT", "10")
        monkeypatch.setenv("UTS_DEFAULT_TIMEOUT", "99")
        importlib.reload(utils.constants)
        assert utils.constants.DEFAULT_TIMEOUT == 99
        importlib.reload(utils.constants)

    def test_uts_prefix_default(self, monkeypatch):
        monkeypatch.setenv("UTS_PREFIX_DEFAULT_TIMEOUT", "50")
        importlib.reload(utils.constants)
        assert utils.constants.DEFAULT_TIMEOUT == 30
        importlib.reload(utils.constants)


class TestParsingFunctions:
    """Test the _as_int, _as_float, _as_bool, _as_str, _as_list functions."""

    def test_as_int_valid(self, monkeypatch):
        monkeypatch.setenv("_T", "42")
        assert utils.constants._as_int("_T", 0) == 42

    def test_as_int_invalid(self, monkeypatch):
        monkeypatch.setenv("_T", "not_a_number")
        assert utils.constants._as_int("_T", 10) == 10

    def test_as_int_missing(self):
        assert utils.constants._as_int("_NONEXISTENT_", 99) == 99

    def test_as_int_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("_T", "  55  ")
        assert utils.constants._as_int("_T", 0) == 55

    def test_as_int_empty_string(self, monkeypatch):
        monkeypatch.setenv("_T", "")
        assert utils.constants._as_int("_T", 7) == 7

    def test_as_int_float_string(self, monkeypatch):
        monkeypatch.setenv("_T", "3.14")
        assert utils.constants._as_int("_T", 1) == 1

    def test_as_float_valid(self, monkeypatch):
        monkeypatch.setenv("_T", "3.14")
        assert utils.constants._as_float("_T", 0.0) == 3.14

    def test_as_float_invalid(self, monkeypatch):
        monkeypatch.setenv("_T", "not_a_float")
        assert utils.constants._as_float("_T", 1.5) == 1.5

    def test_as_float_missing(self):
        assert utils.constants._as_float("_NONEXISTENT_", 2.5) == 2.5

    def test_as_float_int_string(self, monkeypatch):
        monkeypatch.setenv("_T", "42")
        assert utils.constants._as_float("_T", 0.0) == 42.0

    def test_as_bool_true_values(self, monkeypatch):
        for val in ("true", "TRUE", "True", "1", "yes", "on"):
            monkeypatch.setenv("_T", val)
            assert utils.constants._as_bool("_T", False) is True

    def test_as_bool_false_values(self, monkeypatch):
        for val in ("false", "FALSE", "False", "0", "no", "off"):
            monkeypatch.setenv("_T", val)
            assert utils.constants._as_bool("_T", True) is False

    def test_as_bool_invalid(self, monkeypatch):
        monkeypatch.setenv("_T", "maybe")
        assert utils.constants._as_bool("_T", True) is True
        assert utils.constants._as_bool("_T", False) is False

    def test_as_bool_missing(self):
        assert utils.constants._as_bool("_NONEXISTENT_", True) is True
        assert utils.constants._as_bool("_NONEXISTENT_", False) is False

    def test_as_str_valid(self, monkeypatch):
        monkeypatch.setenv("_T", "hello")
        assert utils.constants._as_str("_T", "") == "hello"

    def test_as_str_missing(self):
        assert utils.constants._as_str("_NONEXISTENT_", "fallback") == "fallback"

    def test_as_str_numeric(self, monkeypatch):
        monkeypatch.setenv("_T", "123")
        val = utils.constants._as_str("_T", "")
        assert val == "123"
        assert isinstance(val, str)

    def test_as_list_json_array(self, monkeypatch):
        monkeypatch.setenv("_T", '["x","y","z"]')
        assert utils.constants._as_list("_T", []) == ["x", "y", "z"]

    def test_as_list_json_numbers(self, monkeypatch):
        monkeypatch.setenv("_T", '[1, 2, 3]')
        assert utils.constants._as_list("_T", []) == ["1", "2", "3"]

    def test_as_list_comma_separated(self, monkeypatch):
        monkeypatch.setenv("_T", "a, b, c")
        assert utils.constants._as_list("_T", []) == ["a", "b", "c"]

    def test_as_list_single_value(self, monkeypatch):
        monkeypatch.setenv("_T", "only")
        assert utils.constants._as_list("_T", []) == ["only"]

    def test_as_list_invalid_json(self, monkeypatch):
        monkeypatch.setenv("_T", "{bad json}")
        result = utils.constants._as_list("_T", ["d"])
        assert result == ["{bad json}"]

    def test_as_list_empty_string(self, monkeypatch):
        monkeypatch.setenv("_T", "")
        assert utils.constants._as_list("_T", ["d"]) == ["d"]

    def test_as_list_missing(self):
        assert utils.constants._as_list("_NONEXISTENT_", ["x"]) == ["x"]

    def test_as_list_json_object(self, monkeypatch):
        monkeypatch.setenv("_T", '{"a": 1}')
        result = utils.constants._as_list("_T", ["d"])
        assert result == ['{"a": 1}']

    def test_as_list_empty_json_array(self, monkeypatch):
        monkeypatch.setenv("_T", "[]")
        assert utils.constants._as_list("_T", ["d"]) == []

    def test_as_list_trailing_comma(self, monkeypatch):
        monkeypatch.setenv("_T", "a, b, ")
        assert utils.constants._as_list("_T", []) == ["a", "b"]

    def test_as_list_uts_precedence(self, monkeypatch):
        monkeypatch.setenv("_T", "plain")
        monkeypatch.setenv("UTS__T", "utsprefix")
        assert utils.constants._as_list("_T", []) == ["utsprefix"]
