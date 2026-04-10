# test_ingestion.py
from src.ingestion.simulated_stream import SimulatedStream


def test_run():
    # 1. Initialize the stream
    streamer = SimulatedStream(rate_per_sec=2.0)

    print("🚀 Starting Stream Test (Press Ctrl+C to stop)...")

    # 2. Use the Context Manager you built!
    with streamer as s:
        for i, log in enumerate(s.stream()):
            print(f"Log {i}: {log.event_type} at Aisle {log.aisle} | {log.message}")

            # Stop after 5 logs for the test
            if i >= 4:
                break


if __name__ == "__main__":
    test_run()