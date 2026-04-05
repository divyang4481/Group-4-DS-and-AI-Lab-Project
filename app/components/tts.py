from __future__ import annotations

import time

from app.mechanics.text_to_speech import TTSRuntimeController, TextToSpeech


class TTSComponent:
    def __init__(
        self,
        enabled: bool = False,
        piper_executable: str | None = None,
        voice_model_path: str | None = None,
        voice_config_path: str | None = None,
    ):
        self.enabled = enabled
        self.controller = None
        if enabled and piper_executable and voice_model_path:
            engine = TextToSpeech(
                piper_executable=piper_executable,
                voice_model_path=voice_model_path,
                voice_config_path=voice_config_path,
            )
            engine.load_engine()
            self.controller = TTSRuntimeController(tts_engine=engine, enable_async=True)

    def run(self, ctx):
        if not self.enabled or self.controller is None or not ctx.nav_command:
            return ctx
        start = time.perf_counter()
        result = self.controller.handle_command(ctx.nav_command)
        ctx.metrics["tts_latency_ms"] = (time.perf_counter() - start) * 1000.0
        ctx.metrics.update(result)
        return ctx

    def close(self):
        if self.controller is not None:
            self.controller.stop()
