"""Tests for utils.thread_safety"""
import threading
from utils.thread_safety import (
    document_thread_safety,
    document_method_thread_safety,
    thread_safe,
    get_thread_safety_info,
    generate_thread_safety_report,
    THREAD_SAFETY_NONE,
    THREAD_SAFETY_EXTERNAL,
    THREAD_SAFETY_INTERNAL,
    THREAD_SAFETY_IMMUTABLE,
)


class TestThreadSafetyConstants:
    def test_constants_defined(self):
        assert THREAD_SAFETY_NONE == "none"
        assert THREAD_SAFETY_EXTERNAL == "external"
        assert THREAD_SAFETY_INTERNAL == "internal"
        assert THREAD_SAFETY_IMMUTABLE == "immutable"


class TestDocumentThreadSafety:
    def test_document_class(self):
        @document_thread_safety(THREAD_SAFETY_EXTERNAL, "Test class docs")
        class MyClass:
            """Original docstring."""
            pass

        info = get_thread_safety_info(MyClass)
        assert info["class"] == THREAD_SAFETY_EXTERNAL
        assert "Original docstring." in MyClass.__doc__
        assert "Thread Safety: external" in MyClass.__doc__

    def test_document_class_no_docstring(self):
        @document_thread_safety(THREAD_SAFETY_IMMUTABLE)
        class BareClass:
            pass

        info = get_thread_safety_info(BareClass)
        assert info["class"] == THREAD_SAFETY_IMMUTABLE

    def test_document_method(self):
        class Service:
            @document_method_thread_safety(THREAD_SAFETY_NONE, "Not safe for concurrent use")
            def risky_op(self):
                pass

        info = get_thread_safety_info("Service")
        assert info["risky_op"] == THREAD_SAFETY_NONE

    def test_multiple_methods(self):
        class MultiService:
            @document_method_thread_safety(THREAD_SAFETY_INTERNAL)
            def safe_op(self):
                pass

            @document_method_thread_safety(THREAD_SAFETY_NONE)
            def unsafe_op(self):
                pass

        info = get_thread_safety_info("MultiService")
        assert info["safe_op"] == THREAD_SAFETY_INTERNAL
        assert info["unsafe_op"] == THREAD_SAFETY_NONE


class TestThreadSafeDecorator:
    def test_thread_safe_decorator(self):
        class Counter:
            def __init__(self):
                self.value = 0

            @thread_safe
            def increment(self):
                self.value += 1
                return self.value

        c = Counter()
        # Should have a lock
        assert hasattr(c, "_thread_safe_lock")
        assert isinstance(c._thread_safe_lock, threading.RLock)

        # Basic functionality works
        assert c.increment() == 1
        assert c.increment() == 2

    def test_thread_safe_concurrent_access(self):
        class SharedList:
            def __init__(self):
                self.items = []

            @thread_safe
            def append(self, item):
                self.items.append(item)
                return len(self.items)

        sl = SharedList()
        n_threads = 10
        n_ops = 100

        def worker():
            for i in range(n_ops):
                sl.append(i)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(sl.items) == n_threads * n_ops

    def test_thread_safe_documentation(self):
        @document_thread_safety(THREAD_SAFETY_EXTERNAL, "TestDoc class")
        class TestDoc:
            @thread_safe
            def documented_method(self):
                """Original doc."""
                pass

        info = get_thread_safety_info(TestDoc)
        assert info["class"] == THREAD_SAFETY_EXTERNAL


class TestGenerateReport:
    def test_generate_report(self):
        # Clear registry by resetting internal state
        import utils.thread_safety as ts
        ts._thread_safety_registry.clear()

        @document_thread_safety(THREAD_SAFETY_IMMUTABLE, "Immutable data")
        class ImmutableData:
            pass

        @document_thread_safety(THREAD_SAFETY_EXTERNAL, "External sync")
        class ExternalSync:
            @document_method_thread_safety(THREAD_SAFETY_INTERNAL)
            def safe_call(self):
                pass

        report = generate_thread_safety_report()
        assert "ImmutableData" in report
        assert report["ImmutableData"]["class"] == THREAD_SAFETY_IMMUTABLE
        assert "ExternalSync" in report
        # Note: method documentation uses class name as string, not class reference
        # So it won't appear in the class-keyed report
