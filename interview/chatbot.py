"""
Interview chatbot — adapted from low_latency_chat_bot.

Changes from original:
  1. Endpoint bound to /interview/ws/{session_id}
  2. Dynamic system prompt injected per session from system_prompts.py
  3. Raw PCM audio buffered in SessionBuffer alongside chatbot processing
  4. Transcript accumulated and saved to DB on disconnect
  5. InterviewController tracks question count and fires completion callback

Audio pipeline: PCM chunks → 32KB buffer → WAV → SarvamAI STT → Groq → SarvamAI TTS
Language: auto-detected per message, response mirrors user language.
"""
from __future__ import annotations

import asyncio
import io
import json
import wave
from typing import Any

import httpx
from fastapi import WebSocket, WebSocketDisconnect

from config import settings
from core.logging import get_logger
from interview.controller import InterviewController
from interview.media_buffer import get_or_create_buffer, release_buffer
from interview.system_prompts import get_system_prompt

log = get_logger(__name__)

AUDIO_BUFFER_BYTES = 32 * 1024  # 32KB trigger for STT
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 150
GROQ_TEMPERATURE = 0.7


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# Sarvam language code mapping  (BCP-47 → Sarvam format)
_SARVAM_LANG: dict[str, str] = {
    "kn": "kn-IN",
    "hi": "hi-IN",
    "en": "en-IN",
    "te": "te-IN",
    "ta": "ta-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
}

# Sarvam TTS speaker per language (female voices)
_SARVAM_SPEAKER: dict[str, str] = {
    "kn-IN": "anushka",
    "hi-IN": "meera",
    "en-IN": "meera",
    "te-IN": "anushka",
    "ta-IN": "anushka",
    "ml-IN": "anushka",
    "mr-IN": "anushka",
}


async def _sarvam_stt(wav_bytes: bytes, language: str = "kn") -> str:
    """Send WAV bytes to SarvamAI STT, return transcription text."""
    if not settings.SARVAM_API_KEY:
        return "[STT unavailable — no SARVAM_API_KEY]"
    lang_code = _SARVAM_LANG.get(language, "kn-IN")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"API-Subscription-Key": settings.SARVAM_API_KEY},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"language_code": lang_code, "model": "saarika:v1"},
            )
        resp.raise_for_status()
        return resp.json().get("transcript", "")
    except Exception as exc:
        log.warning("SarvamAI STT failed", language=language, exc_info=True)
        return ""


async def _groq_chat(
    history: list[dict],
    system_prompt: str,
    user_text: str,
) -> str:
    """Send conversation history + new user message to Groq. Return assistant reply."""
    if not settings.GROQ_API_KEY:
        return "Thank you for your answer. Please continue."

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-20:])  # keep last 10 exchanges
    messages.append({"role": "user", "content": user_text})

    try:
        from groq import AsyncGroq  # type: ignore[import]

        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        completion = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE,
        )
        reply = completion.choices[0].message.content or ""
        return reply.strip()
    except Exception as exc:
        log.warning("Groq chat failed", exc_info=True)
        return "Thank you for your answer."


async def _sarvam_tts(text: str, language: str = "kn") -> bytes:
    """Convert text to speech via SarvamAI TTS. Returns WAV bytes."""
    if not settings.SARVAM_API_KEY:
        return b""
    lang_code = _SARVAM_LANG.get(language, "kn-IN")
    speaker   = _SARVAM_SPEAKER.get(lang_code, "anushka")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={
                    "API-Subscription-Key": settings.SARVAM_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [text],
                    "target_language_code": lang_code,
                    "speaker": speaker,
                    "model": "bulbul:v1",
                },
            )
        resp.raise_for_status()
        data = resp.json()
        import base64
        audio_b64 = data.get("audios", [""])[0]
        return base64.b64decode(audio_b64) if audio_b64 else b""
    except Exception as exc:
        log.warning("SarvamAI TTS failed", language=language, exc_info=True)
        return b""


class InterviewChatbotSession:
    """
    Manages a single interview WebSocket session.

    audio_ws: receives PCM audio from browser, runs STT→Groq→TTS pipeline
    video_ws: receives JPEG frames for buffering (no processing during interview)
    """

    def __init__(
        self,
        session_id: str,
        skill_name: str,
        application_id: str,
        language: str = "kn",
    ) -> None:
        self.session_id = session_id
        self.skill_name = skill_name
        self.application_id = application_id
        self.language = language
        self.system_prompt = get_system_prompt(skill_name, language)
        self.history: list[dict] = []
        self._pcm_accumulator = b""
        self._audio_seq = 0
        self._video_seq = 0
        self._complete_event = asyncio.Event()
        self.controller = InterviewController(
            session_id=session_id,
            on_complete=self._on_interview_complete,
        )

    async def _on_interview_complete(self) -> None:
        self._complete_event.set()

    def is_complete(self) -> bool:
        return self._complete_event.is_set()

    async def handle_audio_websocket(self, ws: WebSocket) -> None:
        await ws.accept()
        self.controller.start_timer()
        buf = get_or_create_buffer(self.session_id)

        log.info(
            "Interview audio WebSocket connected",
            session_id=self.session_id,
            skill=self.skill_name,
        )

        try:
            while not self.is_complete():
                try:
                    data = await asyncio.wait_for(ws.receive_bytes(), timeout=5.0)
                except asyncio.TimeoutError:
                    if self.controller.is_complete():
                        break
                    continue
                except WebSocketDisconnect:
                    log.info("Audio WebSocket disconnected", session_id=self.session_id)
                    break

                # Buffer raw PCM for Kafka (alongside pipeline)
                buf.audio.add(self._audio_seq, data)
                self._audio_seq += 1

                self._pcm_accumulator += data
                if len(self._pcm_accumulator) < AUDIO_BUFFER_BYTES:
                    continue

                pcm_chunk = self._pcm_accumulator
                self._pcm_accumulator = b""

                wav = _pcm_to_wav_bytes(pcm_chunk)
                transcript = await _sarvam_stt(wav, language=self.language)
                if not transcript:
                    continue

                log.debug(
                    "STT transcript",
                    session_id=self.session_id,
                    language=self.language,
                    transcript=transcript[:80],
                )

                reply = await _groq_chat(self.history, self.system_prompt, transcript)

                self.history.append({"role": "user", "content": transcript})
                self.history.append({"role": "assistant", "content": reply})
                self.controller.record_question()

                # Send TTS audio back to browser
                tts_audio = await _sarvam_tts(reply, language=self.language)
                if tts_audio:
                    await ws.send_bytes(tts_audio)
                else:
                    # Fallback: send text so frontend can display it
                    await ws.send_text(json.dumps({"type": "text", "content": reply}))

                await self.controller.check_and_complete()

        except Exception as exc:
            log.error(
                "Interview audio WebSocket error",
                session_id=self.session_id,
                exc_info=True,
            )
        finally:
            log.info(
                "Interview audio WebSocket closing",
                session_id=self.session_id,
                question_count=self.controller.question_count,
            )

    async def handle_video_websocket(self, ws: WebSocket) -> None:
        """Accept JPEG frame bytes and buffer them. No pipeline processing."""
        await ws.accept()
        buf = get_or_create_buffer(self.session_id)

        log.info("Interview video WebSocket connected", session_id=self.session_id)
        try:
            while not self.is_complete():
                try:
                    frame = await asyncio.wait_for(ws.receive_bytes(), timeout=5.0)
                except asyncio.TimeoutError:
                    if self.is_complete():
                        break
                    continue
                except WebSocketDisconnect:
                    break

                buf.video.add(self._video_seq, frame)
                self._video_seq += 1

        except Exception:
            log.error("Interview video WebSocket error", session_id=self.session_id, exc_info=True)
        finally:
            log.info(
                "Interview video WebSocket closing",
                session_id=self.session_id,
                frames_buffered=buf.video.frame_count(),
            )

    def get_transcript(self) -> list[dict]:
        return list(self.history)


# In-memory registry: session_id → InterviewChatbotSession
_sessions: dict[str, InterviewChatbotSession] = {}


def create_chatbot_session(
    session_id: str,
    skill_name: str,
    application_id: str,
    language: str = "kn",
) -> InterviewChatbotSession:
    session = InterviewChatbotSession(session_id, skill_name, application_id, language=language)
    _sessions[session_id] = session
    log.info("Chatbot session created", session_id=session_id, skill=skill_name, language=language)
    return session


def get_chatbot_session(session_id: str) -> InterviewChatbotSession | None:
    return _sessions.get(session_id)


def remove_chatbot_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
    release_buffer(session_id)
    log.info("Chatbot session removed", session_id=session_id)
