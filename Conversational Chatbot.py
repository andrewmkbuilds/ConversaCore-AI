import os
import queue
import random
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import google.generativeai as ai

# Prefer an environment variable, but fall back to the inline value for quick experiments.
API_KEY = "YOUR_API_KEY"

ai.configure(api_key=API_KEY)

SUPPORTED_METHOD = "generateContent"
PREFERRED_MODELS: Sequence[str] = (
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-pro",
    "models/gemini-pro-latest",
)
SPARKLE = "\u2728"


def canonical_model(name: Optional[str]) -> str:
    if not name:
        return ""
    return name if name.startswith("models/") else f"models/{name}"


def list_supported_models() -> List[str]:
    try:
        return [
            model.name
            for model in ai.list_models()
            if SUPPORTED_METHOD in getattr(model, "supported_generation_methods", [])
        ]
    except Exception:
        return []


def pick_model(exclude: Optional[Iterable[str]] = None) -> str:
    exclusions: Set[str] = set()
    if exclude:
        if isinstance(exclude, str):
            exclusions.add(exclude)
        else:
            exclusions.update(exclude)
    canonical_exclusions = {canonical_model(name) for name in exclusions}

    available = list_supported_models()
    available_set = {canonical_model(name) for name in available}

    def first_match(candidates: Iterable[str]) -> Optional[str]:
        for candidate in candidates:
            canonical = canonical_model(candidate)
            if canonical in canonical_exclusions:
                continue
            if not available_set or canonical in available_set:
                return canonical
        return None

    preferred_choice = first_match(PREFERRED_MODELS)
    if preferred_choice:
        return preferred_choice

    dynamic_choice = first_match(available)
    if dynamic_choice:
        return dynamic_choice

    return canonical_model(next(iter(PREFERRED_MODELS), "models/gemini-pro"))


class ChatbotApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Gemini Playground")
        self.root.geometry("1280x720")  # 16:9 aspect ratio
        self.root.minsize(960, 540)

        self.sparkle = SPARKLE
        self.model_name = pick_model()
        self.model_display_name = self._friendly_model_name(self.model_name)
        self.model = ai.GenerativeModel(self.model_name)
        self.chat = self.model.start_chat(history=[])

        self.is_sending = False
        self.history: List[Dict[str, str]] = []
        self.theme = "light"
        self.response_queue: "queue.Queue[str]" = queue.Queue()
        self.style = ttk.Style()

        self.persona_prompts: Dict[str, str] = {
            "Friendly": "You are Breezy, an upbeat friend who peppers in emojis and encouragement.",
            "Professional": "You are a concise, insightful assistant focusing on clarity and accuracy.",
            "Creative": "You are Spark, a wildly imaginative storyteller who uses vivid imagery and humor.",
        }
        self.fun_footer = [
            f"{self.sparkle} Fun bonus fact: Sloths can hold their breath longer than dolphins!",
            f"{self.sparkle} High-five! You just unlocked extra brain sparkles.",
            f"{self.sparkle} Here's a sprinkle of inspiration to brighten your day!",
            f"{self.sparkle} Keep the questions coming—curiosity fuels awesome adventures!",
        ]
        self.emoji_codes = [0x1F60A, 0x1F680, 0x1F31F, 0x1F4A1, 0x1F389, 0x1F916, 0x2728, 0x1F340]

        self._build_ui()
        self._apply_theme()

        self.poll_responses()

    def _build_ui(self) -> None:
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        self.chat_frame = tk.Frame(self.main_frame)
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        self.chat_frame.rowconfigure(0, weight=1)
        self.chat_frame.columnconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 12),
            padx=14,
            pady=14,
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew")
        self.chat_display.tag_configure("user", foreground="#2563eb", spacing3=6)
        self.chat_display.tag_configure("assistant", foreground="#16a34a", spacing3=12)
        self.chat_display.tag_configure("system", foreground="#9333ea", font=("Segoe UI", 10, "italic"), spacing3=12)

        self.input_panel = tk.Frame(self.chat_frame)
        self.input_panel.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.input_panel.columnconfigure(0, weight=1)

        self.input_text = tk.Text(
            self.input_panel,
            height=3,
            wrap="word",
            font=("Segoe UI", 12),
            highlightthickness=1,
            relief="groove",
        )
        self.input_text.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_text.bind("<Shift-Return>", self._newline)
        self.input_text.bind("<Return>", self._handle_send_event)

        self.button_panel = tk.Frame(self.input_panel)
        self.button_panel.grid(row=0, column=1, sticky="ns")

        self.send_button = tk.Button(
            self.button_panel,
            text="Send",
            command=self.handle_send,
            width=9,
            height=2,
            font=("Segoe UI", 11, "bold"),
            relief="ridge",
        )
        self.send_button.pack(fill="x", pady=(0, 6))

        tk.Button(
            self.button_panel,
            text="Emoji",
            command=self.insert_emoji,
            width=9,
        ).pack(fill="x")

        self.sidebar = tk.Frame(self.main_frame)
        self.sidebar.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        self.sidebar.columnconfigure(0, weight=1)

        tk.Label(
            self.sidebar,
            text="Conversation Toolkit",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk_label_style = {
            "font": ("Segoe UI", 11, "bold"),
            "anchor": "w",
        }
        ttk_value_style = {
            "font": ("Segoe UI", 11),
            "state": "readonly",
        }

        tk.Label(self.sidebar, text="Persona", **ttk_label_style).grid(row=1, column=0, sticky="ew", pady=(16, 4))
        self.persona_var = tk.StringVar(value="Friendly")
        self.persona_combo = ttk.Combobox(
            self.sidebar,
            textvariable=self.persona_var,
            values=list(self.persona_prompts.keys()),
            **ttk_value_style,
        )
        self.persona_combo.grid(row=2, column=0, sticky="ew")

        tk.Label(self.sidebar, text="Quick prompts", **ttk_label_style).grid(row=3, column=0, sticky="ew", pady=(16, 4))
        self.quick_prompts_frame = tk.Frame(self.sidebar)
        self.quick_prompts_frame.grid(row=4, column=0, sticky="ew")
        self.quick_prompts_frame.columnconfigure(0, weight=1)

        prompts = [
            "Tell me a science fact",
            "Write a tiny story about space cats",
            "Help me plan a productive day",
            "Give me a brain teaser",
        ]
        for prompt in prompts:
            tk.Button(
                self.quick_prompts_frame,
                text=prompt,
                anchor="w",
                command=lambda p=prompt: self.use_quick_prompt(p),
            ).pack(fill="x", pady=2)

        tk.Label(self.sidebar, text="Utilities", **ttk_label_style).grid(row=5, column=0, sticky="ew", pady=(16, 4))
        self.utils_frame = tk.Frame(self.sidebar)
        self.utils_frame.grid(row=6, column=0, sticky="ew")
        self.utils_frame.columnconfigure(0, weight=1)

        tk.Button(self.utils_frame, text="Toggle theme", command=self.toggle_theme).pack(fill="x", pady=2)
        tk.Button(self.utils_frame, text="Summarize chat", command=self.summarize_chat).pack(fill="x", pady=2)
        tk.Button(self.utils_frame, text="Export chat", command=self.export_chat).pack(fill="x", pady=2)
        tk.Button(self.utils_frame, text="Clear chat", command=self.clear_chat).pack(fill="x", pady=2)

        self.status_var = tk.StringVar(value=f"Ready to explore {self.sparkle} (Model: {self.model_display_name})")
        self.stats_var = tk.StringVar(
            value=f"Messages: 0 | Persona: Friendly | Theme: Light | Model: {self.model_display_name}"
        )

        tk.Label(
            self.sidebar,
            textvariable=self.stats_var,
            font=("Segoe UI", 10, "italic"),
            anchor="w",
        ).grid(row=7, column=0, sticky="ew", pady=(20, 4))
        tk.Label(
            self.sidebar,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            anchor="w",
        ).grid(row=8, column=0, sticky="ew")

    def _newline(self, event: tk.Event) -> str:
        self.input_text.insert("insert", "\n")
        return "break"

    def _handle_send_event(self, event: tk.Event) -> str:
        self.handle_send()
        return "break"

    def handle_send(self) -> None:
        if self.is_sending:
            return
        message = self.input_text.get("1.0", "end").strip()
        if not message:
            return
        self.append_chat("You", message, tag="user")
        self.history.append({"role": "user", "content": message})
        self.input_text.delete("1.0", "end")
        self.status_var.set(f"Thinking... {self.sparkle}")
        self.is_sending = True
        self.send_button.config(state="disabled")

        persona = self.persona_var.get()
        threading.Thread(
            target=self._fetch_response,
            args=(message, persona),
            daemon=True,
        ).start()

    def _compose_prompt(self, message: str, persona: str) -> str:
        persona_preamble = self.persona_prompts.get(persona, "")
        transcript = "\n".join(
            f"{entry['role'].capitalize()}: {entry['content']}"
            for entry in self.history[-6:]
        )
        context = f"{transcript}\n" if transcript else ""
        current_time = time.strftime("%A, %B %d %Y %H:%M")
        return f"{persona_preamble}\nCurrent time: {current_time}\n\n{context}User: {message}"

    def _fetch_response(self, message: str, persona: str) -> None:
        prompt = self._compose_prompt(message, persona)
        try:
            response = self.chat.send_message(prompt)
            text = response.text.strip()
            if not text:
                text = "I had a sparkle hiccup and couldn't craft a reply. Try asking again!"
        except Exception as exc:
            if self._attempt_model_recovery(message, persona, exc):
                return
            text = self._build_fallback_response(message, exc)
        else:
            bonus = random.choice(self.fun_footer)
            if random.random() < 0.35:
                text = f"{text}\n\n{bonus}"
        self.response_queue.put(text)

    def _attempt_model_recovery(self, message: str, persona: str, error: Exception) -> bool:
        error_text = str(error).lower()
        if not any(kw in error_text for kw in ("not found", "not supported", "invalid", "unavailable")):
            return False


        tried: Set[str] = {self.model_name}
        last_error: Exception = error
        candidates = self._candidate_models(tried)

        for candidate in candidates:
            try:
                model = ai.GenerativeModel(candidate)
                chat_session = model.start_chat(history=[])
                prompt = self._compose_prompt(message, persona)
                response = chat_session.send_message(prompt)
                text = response.text.strip() if response else ""
                if not text:
                    text = "The new model is ready, but it stayed quiet. Could you ask again?"
                text = (
                    f"{self.sparkle} Quick switch! I hopped from {self.model_display_name} to"
                    f" {self._friendly_model_name(candidate)} to keep things rolling.\n\n{text}"
                )
                self._switch_model(candidate, model=model, chat=chat_session)
                self.response_queue.put(text)
                return True
            except Exception as follow_up_exc:
                last_error = follow_up_exc
                tried.add(candidate)

        self.response_queue.put(self._build_fallback_response(message, last_error))
        return True

    def _candidate_models(self, tried: Iterable[str]) -> List[str]:
        canonical_tried = {canonical_model(name) for name in tried if name}
        available = list_supported_models()
        ordered: List[str] = []

        def add_candidates(candidates: Iterable[str]) -> None:
            for candidate in candidates:
                canonical = canonical_model(candidate)
                if canonical in canonical_tried:
                    continue
                ordered.append(canonical)
                canonical_tried.add(canonical)

        add_candidates(PREFERRED_MODELS)
        add_candidates(available)
        return ordered

    def poll_responses(self) -> None:
        try:
            while True:
                response = self.response_queue.get_nowait()
                self._display_response(response)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_responses)

    def _display_response(self, response: str) -> None:
        self.append_chat("Gemini", response, tag="assistant")
        self.history.append({"role": "assistant", "content": response})
        self.status_var.set(f"Ready for the next adventure {self.sparkle}")
        self.is_sending = False
        self.send_button.config(state="normal")
        self._refresh_stats()

    def append_chat(self, speaker: str, message: str, tag: str) -> None:
        self.chat_display.configure(state="normal")
        timestamp = time.strftime("%H:%M")
        header = f"[{timestamp}] {speaker}:\n"
        self.chat_display.insert("end", header, tag)
        self.chat_display.insert("end", message + "\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.yview_moveto(1.0)

    def insert_emoji(self) -> None:
        emoji = chr(random.choice(self.emoji_codes))
        self.input_text.insert("end", f"{emoji} ")

    def use_quick_prompt(self, prompt: str) -> None:
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", prompt)
        self.handle_send()

    def summarize_chat(self) -> None:
        if not self.history:
            messagebox.showinfo("Summary", "Start a conversation first!")
            return
        recent = "\n".join(f"{item['role']}: {item['content']}" for item in self.history[-10:])
        summary_prompt = (
            "Summarize this conversation in a cheerful tone with 3 bullet points and a short closing note.\n\n"
            f"{recent}"
        )
        try:
            summary = self.model.generate_content(summary_prompt).text.strip()
        except Exception as exc:
            summary = f"Summary unavailable: {exc}"
        self.append_chat("Gemini", summary, tag="system")
        self.history.append({"role": "assistant", "content": summary})
        self._refresh_stats()

    def export_chat(self) -> None:
        if not self.history:
            messagebox.showinfo("Export", "Nothing to export yet. Let's chat first!")
            return
        file_path = filedialog.asksaveasfilename(
            title="Export chat",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("Markdown", "*.md"), ("All Files", "*.*")],
        )
        if not file_path:
            return
        with open(file_path, "w", encoding="utf-8") as handle:
            for entry in self.history:
                handle.write(f"{entry['role'].capitalize()}: {entry['content']}\n\n")
        messagebox.showinfo("Export", f"Chat saved to {file_path}")

    def clear_chat(self) -> None:
        if messagebox.askyesno("Clear chat", "Start fresh? This will erase the visible conversation."):
            self.chat_display.configure(state="normal")
            self.chat_display.delete("1.0", "end")
            self.chat_display.configure(state="disabled")
            self.history.clear()
            self.chat = self.model.start_chat(history=[])
            self.status_var.set(f"All clear! New ideas welcome {self.sparkle}")
            self._refresh_stats()

    def toggle_theme(self) -> None:
        self.theme = "dark" if self.theme == "light" else "light"
        self._apply_theme()

    def _apply_theme(self) -> None:
        if self.theme == "light":
            colors = {
                "bg": "#f8fafc",
                "panel": "#ffffff",
                "text": "#0f172a",
                "accent": "#2563eb",
            }
        else:
            colors = {
                "bg": "#0f172a",
                "panel": "#1e293b",
                "text": "#e2e8f0",
                "accent": "#f97316",
            }

        base_bg = colors["bg"]
        panel_bg = colors["panel"]
        text_color = colors["text"]
        accent = colors["accent"]

        self.root.configure(bg=base_bg)
        self.main_frame.configure(bg=base_bg)
        themed_frames = [
            self.chat_frame,
            self.input_panel,
            self.button_panel,
            self.sidebar,
            self.quick_prompts_frame,
            self.utils_frame,
        ]
        for frame in themed_frames:
            self._paint_frame(frame, panel_bg, text_color, accent)

        self.chat_display.configure(background=panel_bg, foreground=text_color, insertbackground=text_color)
        self.input_text.configure(background=panel_bg, foreground=text_color, insertbackground=text_color)

        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure("Persona.TCombobox", fieldbackground=panel_bg, background=panel_bg, foreground=text_color)
        self.persona_combo.configure(style="Persona.TCombobox")

        self.send_button.configure(bg=accent, fg="#ffffff", activebackground=accent, activeforeground="#ffffff")
        self._refresh_stats()

    def _refresh_stats(self) -> None:
        if hasattr(self, "stats_var"):
            self.stats_var.set(
                f"Messages: {len(self.history)} | Persona: {self.persona_var.get()} | Theme: {self.theme.title()} | Model: {self.model_display_name}"
            )

    def _switch_model(
        self,
        new_model: str,
        *,
        model: Optional[ai.GenerativeModel] = None,
        chat: Optional[Any] = None,
    ) -> None:
        canonical = canonical_model(new_model)
        self.model_name = canonical
        self.model_display_name = self._friendly_model_name(canonical)
        self.model = model if model is not None else ai.GenerativeModel(canonical)
        self.chat = chat if chat is not None else self.model.start_chat(history=[])
        if hasattr(self, "status_var"):
            self.status_var.set(f"Switched to {self.model_display_name} {self.sparkle} Ready when you are!")
        self._refresh_stats()

    @staticmethod
    def _friendly_model_name(model_name: str) -> str:
        return model_name.split("/", 1)[-1]

    def _paint_frame(self, frame: tk.Widget, bg: str, text_color: str, accent: str) -> None:
        try:
            frame.configure(bg=bg)
        except tk.TclError:
            return
        for child in frame.winfo_children():
            if isinstance(child, tk.Frame):
                self._paint_frame(child, bg, text_color, accent)
            elif isinstance(child, tk.Label):
                child.configure(bg=bg, fg=text_color)
            elif isinstance(child, tk.Button):
                if child is self.send_button:
                    continue
                child.configure(bg=bg, fg=text_color, activebackground=accent, activeforeground="#ffffff")
            elif isinstance(child, ttk.Combobox):
                child.configure(style="Persona.TCombobox")

    def _build_fallback_response(self, message: str, exc: Exception) -> str:
        playful = [
            "My crystal ball fogged up. Can we try that again?",
            "Whoops! I tangled my circuits while thinking. Mind rephrasing?",
            "Even AI needs a snack break sometimes. Let's take another swing!",
        ]
        return (
            f"I hit a glitch while crafting a reply ({exc}). {random.choice(playful)}\n"
            "In the meantime, here's a fun tip: Ask me for a surprise poem!"
        )

    def poll_persona_change(self) -> None:
        self._refresh_stats()
        self.root.after(500, self.poll_persona_change)


def main() -> None:
    root = tk.Tk()
    app = ChatbotApp(root)
    app.poll_persona_change()
    root.mainloop()


if __name__ == "__main__":
    main()


