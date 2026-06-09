"""
Voice handler for the Robot LfD system.

Handles:
- Speech-to-text using local Whisper
- Intent extraction using rule-based keyword matching
- Task similarity matching using word overlap
- Text-to-speech using pyttsx3

Dependencies:
    pip install openai-whisper pyttsx3 audio-recorder-streamlit
    sudo apt-get install espeak  # required by pyttsx3 on Linux
"""

import os
import re
import tempfile
import threading
from pathlib import Path

import whisper

# Intent rules
# Each intent is a list of trigger phrases.

INTENT_RULES = {
    "stop_recording": [
        "stop recording", "stop", "done", "finished", "that's it",
        "thats it", "end recording", "i'm done", "im done",
    ],
    "start_recording": [
        "start recording", "begin recording", "start demo",
        "start demonstration", "record", "let's record", "lets record",
    ],
    "start_training": [
        "start training", "train", "train the model", "train the policy",
        "begin training", "learn from demos", "learn",
    ],
    "execute_task": [
        "show me", "execute", "run the policy", "perform the task",
        "show what you can do", "demonstrate", "run",
    ],
    "list_tasks": [
        "what tasks", "which tasks", "list tasks", "what do you know",
        "what can you do", "show tasks",
    ],
    "go_home": [
        "go home", "home position", "reset position", "move home",
    ],
    "confirm_yes": [
        "yes", "yeah", "yep", "sure", "okay", "ok", "please",
        "yes please", "go ahead", "do it",
    ],
    "confirm_no": [
        "no", "nope", "cancel", "skip", "don't", "do not",
    ],
    "create_task": [
        "teach you", "teach the robot", "new task", "i want to teach",
        "learn a new task", "create task", "add task",
    ],
    "greeting": [
        "hello", "hallo", "hi", "hey", "good morning", "good afternoon",
    ],
}

# Words to strip when extracting a task name
STOPWORDS = {
    "i", "want", "to", "teach", "you", "the", "robot", "a", "an", "new",
    "task", "called", "named", "please", "can", "how", "do", "would",
    "learn", "create", "add", "let", "me", "show", "like", "make",
}


class VoiceHandler:
    """
    Stateless voice processing handler.
    Conversation history and pending actions are managed externally
    in Streamlit session state.
    """

    _whisper_model = None  # class-level cache — loads once per process
    _tts_engine    = None
    _tts_lock      = threading.Lock()

    def __init__(self, demos_dir: Path):
        self.demos_dir = demos_dir

    # Speech-to-text

    @classmethod
    def _get_whisper(cls):
        if cls._whisper_model is None:
            cls._whisper_model = whisper.load_model("base")
        return cls._whisper_model

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe raw audio bytes to text using local Whisper."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            model = self._get_whisper()
            result = model.transcribe(tmp_path)
            return result["text"].strip()
        finally:
            os.unlink(tmp_path)

    # Text-to-speech

    @classmethod
    def _get_tts(cls):
        if cls._tts_engine is None:
            import pyttsx3
            cls._tts_engine = pyttsx3.init()
            cls._tts_engine.setProperty("rate", 150)
            cls._tts_engine.setProperty("volume", 1.0)
        return cls._tts_engine

    @classmethod
    def speak(cls, text: str):
        """Speak text via TTS in a background thread (non-blocking)."""
        def _run():
            with cls._tts_lock:
                try:
                    engine = cls._get_tts()
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass  # TTS failure should never crash the UI

        threading.Thread(target=_run, daemon=True).start()

    # Task helpers

    def get_available_tasks(self) -> list[str]:
        """Return task names from the demos folder."""
        if not self.demos_dir.exists():
            return []
        return sorted([d.name for d in self.demos_dir.iterdir() if d.is_dir()])

    def find_similar_task(self, task_name: str) -> str | None:
        """
        Return the most similar existing task name using word overlap, or None.
        """
        tasks = self.get_available_tasks()
        if not tasks:
            return None

        query_words = set(task_name.lower().replace("_", " ").split())
        best_task, best_score = None, 0

        for t in tasks:
            t_words = set(t.lower().replace("_", " ").split())
            overlap = len(query_words & t_words)
            if overlap > best_score:
                best_score = overlap
                best_task  = t

        return best_task if best_score > 0 else None

    # Intent extraction

    @staticmethod
    def _detect_intent(text: str) -> str:
        """Return the first matching intent key, or 'none'."""
        lower = text.lower()
        for intent, phrases in INTENT_RULES.items():
            for phrase in phrases:
                if phrase in lower:
                    return intent
        return "none"

    @staticmethod
    def _extract_task_name(text: str) -> str | None:
        """
        Extract a task name from speech like:
        'teach you pick up the mug' → 'pick_up_the_mug'
        'new task called stack the blocks' → 'stack_the_blocks'
        """
        lower = text.lower()

        # Try "called X" or "named X" pattern first
        match = re.search(r"(?:called|named)\s+(.+)", lower)
        if match:
            raw = match.group(1).strip()
        else:
            # Strip stopwords from the full utterance
            words = [w for w in lower.split() if w not in STOPWORDS]
            raw   = " ".join(words).strip()

        if not raw:
            return None

        # Convert to snake_case and remove punctuation
        task_name = re.sub(r"[^\w\s]", "", raw).strip().replace(" ", "_")
        return task_name if task_name else None

    # Main process method

    def process(
        self,
        user_text: str,
        pending: dict | None = None,
    ) -> dict:
        """
        Parse user_text and return a result dict:
            message      – response to speak/display
            action       – action string (matches INTENT_RULES keys or "none")
            task_name    – extracted snake_case task name or None
            confirmed    – True if user confirmed a pending action
            similar_task – name of closest existing task or None
        """
        intent    = self._detect_intent(user_text)
        task_name = None
        confirmed = False
        message   = ""

        # Handle pending confirmation
        if pending:
            if intent == "confirm_yes":
                confirmed = True
                # Carry through the pending action
                intent    = pending.get("action", "none")
                task_name = pending.get("task_name")
                message   = "Got it, proceeding."
            elif intent == "confirm_no":
                message   = "Okay, cancelled."
                intent    = "none"
            # If neither yes nor no, fall through to normal intent handling

        # Generate response message
        if not message:
            if intent == "create_task":
                task_name = self._extract_task_name(user_text)
                # Reject single generic words that are not real task names
                if task_name and len(task_name) <= 4 and "_" not in task_name:
                    task_name = None
                if task_name:
                    similar = self.find_similar_task(task_name)
                    if similar and similar != task_name:
                        message = (
                            f"I found a similar existing task called '{similar}'. "
                            f"Would you like to see it first?"
                        )
                        # Return early — wait for confirmation before acting
                        return {
                            "message":     message,
                            "action":      "create_task",
                            "task_name":   task_name,
                            "confirmed":   False,
                            "similar_task": similar,
                            "awaiting_confirmation": True,
                            "confirmation_action": {
                                "action":    "execute_task",
                                "task_name": similar,
                            },
                        }
                    else:
                        message = (
                            f"Creating new task '{task_name}'. "
                            f"Say 'start recording' when you are ready."
                        )
                else:
                    message = (
                        "What would you like to call the new task? "
                        "For example, say 'new task called pick up the mug'."
                    )

            elif intent == "start_recording":
                message = "Starting the simulation. Perform the task, then say 'stop recording'."

            elif intent == "stop_recording":
                message = "Recording stopped. Go to the Record tab to process your demos."

            elif intent == "start_training":
                message = "Starting training. This will take a few minutes."

            elif intent == "execute_task":
                message = "Running the policy now. Watch the simulation window."

            elif intent == "list_tasks":
                tasks = self.get_available_tasks()
                if tasks:
                    message = f"I know these tasks: {', '.join(tasks)}."
                else:
                    message = "No tasks recorded yet. Teach me one first."

            elif intent == "go_home":
                message = "Moving to home position."

            elif intent == "greeting":
                message = (
                    "Hello! I am ready to learn. "
                    "You can teach me a new task, ask me to record a demonstration, or show you what I know."
                )

            elif intent == "confirm_yes":
                confirmed = True
                message   = "Got it."

            elif intent == "confirm_no":
                message = "Okay, no problem."

            else:
                message = (
                    "I did not understand that. You can say things like "
                    "'teach you a new task', 'start recording', or 'show me what you can do'."
                )

        similar = self.find_similar_task(task_name) if task_name else None

        return {
            "message":               message,
            "action":                intent,
            "task_name":             task_name,
            "confirmed":             confirmed,
            "similar_task":          similar,
            "awaiting_confirmation": False,
            "confirmation_action":   None,
        }
