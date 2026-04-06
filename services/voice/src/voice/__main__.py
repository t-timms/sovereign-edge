"""Entry point for the voice service.

Usage:
    uv run python -m voice
    uv run python -m voice --no-wake           # always listening, no wake word
    uv run python -m voice --whisper small.en  # higher accuracy STT
    uv run python -m voice --check-deps        # verify all deps are installed
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

from observability.logging import get_logger

logger = get_logger(__name__, component="voice")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sovereign Edge voice assistant — wake word → STT → LLM → TTS"
    )
    parser.add_argument(
        "--no-wake",
        action="store_true",
        help="Skip wake word detection — always listen (dev mode)",
    )
    parser.add_argument(
        "--wake-model",
        type=str,
        default=None,
        help="Path to custom OpenWakeWord .onnx model (default: bundled)",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default="en_US-lessac-medium",
        help="Piper voice model name (default: en_US-lessac-medium)",
    )
    parser.add_argument(
        "--whisper",
        type=str,
        default="base.en",
        help="faster-whisper model size: tiny.en, base.en, small.en (default: base.en)",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="voice_default",
        help="LangGraph conversation thread ID (default: voice_default)",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check that optional voice dependencies are installed and exit",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Emit structured JSON logs",
    )
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> None:
    from career.expert import CareerExpert
    from creative.expert import CreativeExpert
    from goals.expert import GoalExpert
    from intelligence.expert import IntelligenceExpert
    from orchestrator.main import Orchestrator
    from spiritual.expert import SpiritualExpert

    from voice.pipeline import VoicePipeline

    orch = Orchestrator()
    for expert in (
        SpiritualExpert(),
        CareerExpert(),
        IntelligenceExpert(),
        CreativeExpert(),
        GoalExpert(),
    ):
        orch.register(expert)
    await orch.start()

    pipeline = VoicePipeline(
        orchestrator=orch,
        wake_model_path=args.wake_model,
        tts_voice=args.tts_voice,
        whisper_model=args.whisper,
        thread_id=args.thread_id,
    )

    if args.check_deps:
        deps = pipeline.check_deps()
        print("\nVoice dependency check:")
        all_ok = True
        for name, available in deps.items():
            status = "✓" if available else "✗ NOT INSTALLED"
            print(f"  {name:<20} {status}")
            if not available:
                all_ok = False
        if not all_ok:
            print("\nInstall missing deps in mlenv:")
            print("  pip install openwakeword sounddevice faster-whisper piper-tts scipy")
        sys.exit(0 if all_ok else 1)

    # Register signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def _shutdown(sig: signal.Signals) -> None:
        logger.info("voice.shutdown", signal=sig.name)
        pipeline.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown, sig)
        except (NotImplementedError, OSError):
            pass  # Windows doesn't support add_signal_handler for all signals

    use_wake = not args.no_wake
    try:
        await pipeline.run_forever(use_wake_word=use_wake)
    finally:
        await orch.stop()


def main() -> None:
    from observability.logging import setup_logging

    args = _parse_args()
    setup_logging(debug=False)
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
