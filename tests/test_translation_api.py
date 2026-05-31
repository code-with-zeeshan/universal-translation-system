"""
Tests for integration.translation_api.

Relies on conftest.py for all heavy dependency mocks (prometheus_client, yaml, etc.).
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from integration.translation_api import (
    translate,
    translate_async,
    translate_batch_async,
    evaluate,
    evaluate_async,
    integrate_full_pipeline,
    integrate_full_pipeline_async,
)


class TestTranslateFunction(unittest.TestCase):
    """Test translate function."""

    def setUp(self):
        self.system = MagicMock()
        self.system.encoder = MagicMock()
        self.system.decoder = MagicMock()
        self.system.evaluator = MagicMock()
        self.system.vocab_manager = MagicMock()
        self.system.vocab_manager.language_to_pack = {'es': 'latin'}
        self.system.vocab_manager.get_vocab_for_pair.return_value = MagicMock()

    @patch('integration.translation_api.InputValidator')
    def test_translate_calls_validator(self, mock_validator):
        mock_validator.validate_text_input.return_value = "hello"
        mock_validator.validate_language_code.return_value = True
        self.system.evaluator.translate.return_value = "Hola"

        result = translate(self.system, "hello", "es", "es")
        mock_validator.validate_text_input.assert_called_once_with("hello", max_length=5000)
        self.assertEqual(result, "Hola")

    @patch('integration.translation_api.InputValidator')
    def test_translate_invalid_source_lang(self, mock_validator):
        mock_validator.validate_text_input.return_value = "hello"
        mock_validator.validate_language_code.side_effect = [False, True]
        with self.assertRaises(ValueError) as ctx:
            translate(self.system, "hello", "invalid", "en")
        self.assertIn("source language", str(ctx.exception).lower())

    @patch('integration.translation_api.InputValidator')
    def test_translate_invalid_target_lang(self, mock_validator):
        mock_validator.validate_text_input.return_value = "hello"
        mock_validator.validate_language_code.side_effect = [True, False]
        with self.assertRaises(ValueError) as ctx:
            translate(self.system, "hello", "es", "invalid")
        self.assertIn("target language", str(ctx.exception).lower())

    @patch('integration.translation_api.InputValidator')
    def test_translate_models_not_initialized(self, mock_validator):
        mock_validator.validate_text_input.return_value = "hello"
        mock_validator.validate_language_code.return_value = True
        self.system.encoder = None
        self.system.decoder = None
        with self.assertRaises(RuntimeError):
            translate(self.system, "hello", "es", "en")

    @patch('integration.translation_api.InputValidator')
    def test_translate_evaluator_not_initialized(self, mock_validator):
        mock_validator.validate_text_input.return_value = "hello"
        mock_validator.validate_language_code.return_value = True
        self.system.evaluator = None
        with self.assertRaises(RuntimeError):
            translate(self.system, "hello", "es", "en")

    @patch('integration.translation_api.InputValidator')
    def test_translate_with_domain(self, mock_validator):
        mock_validator.validate_text_input.return_value = "hello"
        mock_validator.validate_language_code.return_value = True
        self.system.evaluator.translate.return_value = "Medical translation"

        result = translate(self.system, "hello", "es", "en", domain="medical")
        self.system.encoder.load_language_adapter.assert_called_once()
        self.assertEqual(result, "Medical translation")

    @patch('integration.translation_api.InputValidator')
    def test_translate_vocab_fallback(self, mock_validator):
        mock_validator.validate_text_input.return_value = "hello"
        mock_validator.validate_language_code.return_value = True
        self.system.vocab_manager.get_vocab_for_pair.side_effect = [
            Exception("Not found"), MagicMock()
        ]
        self.system.evaluator.translate.return_value = "Hola"

        result = translate(self.system, "hello", "es", "en", domain="medical")
        self.assertEqual(result, "Hola")

    @patch('integration.translation_api.InputValidator')
    def test_translate_vocab_fallback_re_raise(self, mock_validator):
        mock_validator.validate_text_input.return_value = "hello"
        mock_validator.validate_language_code.return_value = True
        self.system.vocab_manager.get_vocab_for_pair.side_effect = Exception("General error")

        with self.assertRaises(Exception):
            translate(self.system, "hello", "es", "en")


class TestTranslateAsync(unittest.TestCase):
    """Test translate_async function."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.system = MagicMock()
        self.system.executor = MagicMock()
        self.system.translate = MagicMock(return_value="Hola")

    def tearDown(self):
        self.loop.close()

    def _make_future(self, result):
        fut = self.loop.create_future()
        fut.set_result(result)
        return fut

    def test_translate_async(self):
        self.system.executor.submit.return_value = self._make_future("Hola")

        async def run():
            return await translate_async(self.system, "hello", "es", "en")

        result = self.loop.run_until_complete(run())
        self.assertEqual(result, "Hola")

    def test_translate_async_metrics_recorded(self):
        from integration.translation_api import translation_counter, translation_duration
        translation_counter.reset_mock()
        translation_duration.reset_mock()

        self.system.executor.submit.return_value = self._make_future("Hola")

        async def run():
            return await translate_async(self.system, "hello", "es", "en")

        self.loop.run_until_complete(run())
        translation_counter.labels.assert_called_with(source_lang='es', target_lang='en')


class TestTranslateBatchAsync(unittest.TestCase):
    """Test translate_batch_async function."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.system = MagicMock()
        self.system.executor = MagicMock()
        self.system.translate = MagicMock(return_value="hello")

        async def mock_translate_async(text, source_lang, target_lang):
            return "hello"
        self.system.translate_async = mock_translate_async

    def tearDown(self):
        self.loop.close()

    def test_translate_batch_async(self):
        async def run():
            return await translate_batch_async(
                self.system, ["hello", "world"], "es", "en", max_concurrent=5
            )

        result = self.loop.run_until_complete(run())


class TestEvaluateFunction(unittest.TestCase):
    """Test evaluate function."""

    def test_no_evaluator(self):
        system = MagicMock()
        system.evaluator = None
        result = evaluate(system, "/tmp/test.txt")
        self.assertIsNone(result)

    def test_with_evaluator(self):
        system = MagicMock()
        system.evaluator = MagicMock()
        system.evaluator.evaluate_file.return_value = {"bleu": 42.0}
        result = evaluate(system, "/tmp/test.txt")
        self.assertEqual(result, {"bleu": 42.0})

    def test_with_output_file(self):
        system = MagicMock()
        system.evaluator = MagicMock()
        system.evaluator.evaluate_file.return_value = {"bleu": 42.0}
        result = evaluate(system, "/tmp/test.txt", output_file="/tmp/report.json")
        system.evaluator.create_evaluation_report.assert_called_once_with(
            {"bleu": 42.0}, "/tmp/report.json"
        )


class TestEvaluateAsync(unittest.TestCase):
    """Test evaluate_async function."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_evaluate_async(self):
        system = MagicMock()
        system.executor = MagicMock()
        system.evaluate = MagicMock(return_value={"bleu": 42.0})
        fut = self.loop.create_future()
        fut.set_result({"bleu": 42.0})
        system.executor.submit.return_value = fut

        async def run():
            return await evaluate_async(system, "/tmp/test.txt")

        result = self.loop.run_until_complete(run())
        self.assertEqual(result, {"bleu": 42.0})


class TestIntegrateFullPipeline(unittest.TestCase):
    """Test integrate_full_pipeline."""

    @patch('integration.translation_api.integrate_full_pipeline_async')
    @patch('integration.translation_api.asyncio')
    def test_integrate_full_pipeline(self, mock_asyncio, mock_async):
        mock_asyncio.run = MagicMock()
        result = integrate_full_pipeline(config_file="/tmp/config.yaml")
        mock_asyncio.run.assert_called_once()


class TestIntegrateFullPipelineAsync(unittest.TestCase):
    """Test integrate_full_pipeline_async."""

    @patch('integration.translation_api.SystemConfig')
    @patch('integration.translation_api.UniversalTranslationSystem')
    @patch('builtins.open')
    def test_success(
        self, mock_open, mock_system_cls, mock_config_cls
    ):
        from integration.translation_api import yaml
        yaml.safe_load.return_value = {"data_dir": "/tmp/data"}
        mock_config_cls.return_value = MagicMock()
        mock_system = MagicMock()
        mock_system.initialize_all_systems.return_value = True

        async def mock_health():
            return {"status": "healthy"}
        mock_system.health_check_async = mock_health
        mock_system_cls.return_value = mock_system

        result = asyncio.run(integrate_full_pipeline_async(config_file="/tmp/config.yaml"))
        self.assertIs(result, mock_system)

    @patch('integration.translation_api.SystemConfig')
    @patch('integration.translation_api.UniversalTranslationSystem')
    @patch('builtins.open')
    def test_failed_init(self, mock_open, mock_system_cls, mock_config_cls):
        mock_system = MagicMock()
        mock_system.initialize_all_systems.return_value = False
        mock_system_cls.return_value = mock_system

        result = asyncio.run(integrate_full_pipeline_async())
        self.assertIsNone(result)

    @patch('integration.translation_api.SystemConfig')
    @patch('integration.translation_api.UniversalTranslationSystem')
    @patch('builtins.open')
    def test_file_not_found_fallback(
        self, mock_open, mock_system_cls, mock_config_cls
    ):
        mock_open.side_effect = FileNotFoundError("No file")
        mock_config_cls.return_value = MagicMock()
        mock_system = MagicMock()
        mock_system.initialize_all_systems.return_value = True

        async def mock_health():
            return {"status": "healthy"}
        mock_system.health_check_async = mock_health
        mock_system_cls.return_value = mock_system

        result = asyncio.run(integrate_full_pipeline_async(config_file="/tmp/config.yaml"))
        self.assertIs(result, mock_system)

    @patch('integration.translation_api.SystemConfig')
    @patch('integration.translation_api.UniversalTranslationSystem')
    @patch('builtins.open')
    def test_yaml_error_fallback(
        self, mock_open, mock_system_cls, mock_config_cls
    ):
        mock_open.return_value.__enter__.return_value = MagicMock()
        mock_config_cls.return_value = MagicMock()
        mock_system = MagicMock()
        mock_system.initialize_all_systems.return_value = True

        async def mock_health():
            return {"status": "healthy"}
        mock_system.health_check_async = mock_health
        mock_system_cls.return_value = mock_system

        from integration.translation_api import yaml
        with patch.object(yaml, 'safe_load') as mock_safe_load:
            mock_safe_load.side_effect = yaml.YAMLError("Parse error")
            result = asyncio.run(integrate_full_pipeline_async(config_file="/tmp/config.yaml"))
            self.assertIs(result, mock_system)


class TestPatchedMethods(unittest.TestCase):
    """Test that methods are patched onto UniversalTranslationSystem."""

    @patch('integration.translation_api.UniversalTranslationSystem')
    def test_methods_patched(self, mock_system_cls):
        from integration.translation_api import (
            translate as orig_translate,
            translate_async as orig_translate_async,
            translate_batch_async as orig_batch_async,
            evaluate as orig_evaluate,
            evaluate_async as orig_evaluate_async,
        )
        self.assertTrue(callable(orig_translate))
        self.assertTrue(callable(orig_translate_async))
        self.assertTrue(callable(orig_batch_async))
        self.assertTrue(callable(orig_evaluate))
        self.assertTrue(callable(orig_evaluate_async))


class TestFunctionSignatures(unittest.TestCase):
    """Test function signatures exist."""

    def test_all_callable(self):
        self.assertTrue(callable(integrate_full_pipeline))
        self.assertTrue(callable(integrate_full_pipeline_async))
        self.assertTrue(callable(translate))
