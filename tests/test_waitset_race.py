"""WaitSet must tolerate concurrent attach/detach while another thread waits.

Reports H1/B3: the node's processing thread calls WaitSet.wait() while the
messaging thread (link wiring, expression-slot subscribe/unsubscribe) attaches
and detaches listeners. Without internal locking the waiter's view of the
listener set races the mutation, which can drop a wakeup or, at the iox2 layer,
wait on a torn-down listener. The lock serializes mutation against the waiter's
snapshot (released before every blocking wait, so churn never stalls).

The race is largely GIL-masked, so this is a concurrency-safety + liveness
guard: under heavy churn the waiter must not crash or deadlock, and a fresh
notification afterwards must still wake it. Wake/timeout correctness itself is
covered by test_transport's waitset tests.
"""
import threading
import time

from goofi.transport import WaitSet, open_publisher, open_subscriber, set_instance_id


def test_waitset_concurrent_churn_is_safe_and_stays_live():
    set_instance_id("waitset-race")

    # A real channel whose listener we keep attached the whole time.
    pub, notifier = open_publisher("waitset-race-live", in_process=True)
    live_sub, live_listener = open_subscriber("waitset-race-live", in_process=True)

    ws = WaitSet()
    ws.attach(live_listener)
    churn_listeners = [open_subscriber(f"waitset-race-{i}", in_process=True)[1] for i in range(8)]

    errors: list = []
    stop = threading.Event()

    def churn():
        try:
            for _ in range(4000):
                for listener in churn_listeners:
                    ws.detach(listener)
                    ws.attach(listener)
                if stop.is_set():
                    return
        except Exception as e:
            errors.append(("churn", repr(e)))
        finally:
            stop.set()

    t = threading.Thread(target=churn)
    t.start()
    try:
        while not stop.is_set():
            try:
                ws.wait(0.0)
            except Exception as e:
                errors.append(("wait", repr(e)))
                break
    finally:
        stop.set()
        t.join(timeout=30)

    assert not t.is_alive(), "churn deadlocked under the WaitSet lock"
    assert not errors, f"WaitSet raced under concurrent attach/detach: {errors}"

    # Liveness: a fresh notification still wakes the waiter after all the churn.
    pub.send(b"x")
    notifier.notify()
    deadline = time.time() + 2.0
    fired = []
    while time.time() < deadline and live_listener not in fired:
        fired = ws.wait(0.1)
    assert live_listener in fired, "waiter went deaf after concurrent churn"
