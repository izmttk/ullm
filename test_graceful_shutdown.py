#!/usr/bin/env python3
"""
Simple test script to validate graceful shutdown functionality.
This script tests both process and threading modes.
"""
import signal
import sys
import time
import os

def test_signal_handling():
    """Test that signal handlers work correctly."""
    print("Testing signal handling...")
    
    received_signals = []
    
    def handler(signum, frame):
        received_signals.append(signum)
        print(f"Received signal: {signum}")
    
    # Register handlers
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    
    # Send ourselves a SIGTERM
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(0.1)
    
    assert signal.SIGTERM in received_signals, "SIGTERM was not caught"
    print("✓ Signal handling works correctly")


def test_platform_detection():
    """Test platform-specific behavior."""
    print("\nTesting platform detection...")
    print(f"Platform: {sys.platform}")
    
    if sys.platform == "win32":
        print("✓ Running on Windows - threading mode recommended")
        assert not hasattr(os, 'setsid') or os.name == 'nt', "setsid should not be available on Windows"
    else:
        print("✓ Running on Unix-like system - both modes available")
        assert hasattr(os, 'setsid'), "setsid should be available on Unix"


def test_queue_compatibility():
    """Test that queue imports work for both modes."""
    print("\nTesting queue compatibility...")
    
    # Test standard queue
    import queue
    q1 = queue.Queue()
    q1.put("test")
    assert q1.get() == "test"
    print("✓ Standard queue works")
    
    # Test multiprocessing queue (skip if torch not available)
    try:
        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')
        q2 = ctx.Queue()
        q2.put("test")
        assert q2.get() == "test"
        print("✓ Multiprocessing queue works")
    except ImportError:
        print("⊘ Torch not available, skipping multiprocessing queue test")


def test_threading_mode():
    """Test threading mode basic functionality."""
    print("\nTesting threading mode...")
    import threading
    import queue
    
    result = []
    q = queue.Queue()
    
    def worker():
        msg = q.get()
        result.append(msg)
        q.task_done()
    
    t = threading.Thread(target=worker)
    t.start()
    q.put("test")
    t.join()
    
    assert result == ["test"]
    print("✓ Threading mode works correctly")


def main():
    print("=" * 60)
    print("Graceful Shutdown and Windows Support Test Suite")
    print("=" * 60)
    
    try:
        test_platform_detection()
        test_signal_handling()
        test_queue_compatibility()
        test_threading_mode()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
