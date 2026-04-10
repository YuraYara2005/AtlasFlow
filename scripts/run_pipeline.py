# scripts/run_pipeline.py
import sys
import logging
from src.ingestion.simulated_stream import SimulatedStream
from src.preprocessing.log_processor import LogProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    print("🚀 Starting AI Warehouse Pipeline...")

    # 1. Set up the stream (fast rate for testing)
    stream = SimulatedStream(rate_per_sec=10.0, seed=42)

    # 2. Set up the processor (window size of 5 logs)
    processor = LogProcessor(window_size=5)

    # 3. Run the integration!
    with stream as active_stream:
        for i, log in enumerate(active_stream.stream()):
            tensor = processor.process(log)

            if tensor is not None:
                print(f"\nWindow Complete at log {i}!")
                print(f"Tensor Shape: {tensor.shape}")
                print(f"Tensor Data:\n{tensor}")
                break  # Stop after the first successful tensor!


if __name__ == "__main__":
    main()