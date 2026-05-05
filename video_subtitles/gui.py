"""Desktop GUI for the subtitle generator."""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, Listbox, StringVar, Text, Tk, filedialog, messagebox
from tkinter import ttk

current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))
sys.path.insert(1, str(current_dir.parent.resolve()))

import config
from app_pipeline import ProcessingOptions, collect_tasks, get_json_files, get_video_files, run_batch

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    TkinterDnD = None

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".wmv"}


def split_drop_paths(raw: str) -> list[Path]:
    if not raw:
        return []
    paths = []
    token = []
    in_brace = False
    for char in raw:
        if char == "{":
            in_brace = True
            token = []
            continue
        if char == "}":
            in_brace = False
            if token:
                paths.append(Path("".join(token)))
                token = []
            continue
        if char == " " and not in_brace:
            if token:
                paths.append(Path("".join(token)))
                token = []
            continue
        token.append(char)
    if token:
        paths.append(Path("".join(token)))
    return paths


class SubtitleGui:
    def __init__(self) -> None:
        root_cls = TkinterDnD.Tk if TkinterDnD else Tk
        self.root = root_cls()
        self.root.title("Whisper Subtitle GUI")
        self.root.geometry("980x720")

        self.task_paths: list[Path] = []
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.model_var = StringVar(value=config.MODEL_NAME)
        self.mode_var = StringVar(value=config.DEFAULT_MODE)
        self.skip_translate_var = BooleanVar(value=config.SKIP_TRANSLATION)
        self.translate_only_var = BooleanVar(value=False)
        self.rescue_var = BooleanVar(value=config.ENABLE_GAP_RESCUE)
        self.status_var = StringVar(value="대기 중")

        self._build_layout()
        self._refresh_drop_state()
        self.root.after(150, self._flush_logs)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=2)

        header = ttk.Frame(self.root, padding=16)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Whisper 자막 생성기", font=("Malgun Gothic", 16, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="영상 또는 JSON을 추가하고 옵션을 선택한 뒤 실행합니다.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        options = ttk.LabelFrame(self.root, text="실행 옵션", padding=16)
        options.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 12))
        for col in range(4):
            options.columnconfigure(col, weight=1 if col in (1, 3) else 0)

        ttk.Label(options, text="Whisper 모델").grid(row=0, column=0, sticky="w")
        model_values = ["tiny", "base", "small", "medium", "large-v3"]
        ttk.Combobox(options, textvariable=self.model_var, values=model_values, state="normal").grid(
            row=0, column=1, sticky="ew", padx=(8, 16)
        )

        ttk.Label(options, text="분석 모드").grid(row=0, column=2, sticky="w")
        ttk.Combobox(options, textvariable=self.mode_var, values=["movie", "music"], state="readonly").grid(
            row=0, column=3, sticky="ew", padx=(8, 0)
        )

        ttk.Checkbutton(
            options,
            text="번역 건너뛰기",
            variable=self.skip_translate_var,
            command=self._sync_translate_mode,
        ).grid(row=1, column=0, sticky="w", pady=(12, 0))
        ttk.Checkbutton(
            options,
            text="JSON 번역만 수행",
            variable=self.translate_only_var,
            command=self._sync_translate_mode,
        ).grid(row=1, column=1, sticky="w", pady=(12, 0))
        ttk.Checkbutton(
            options,
            text="Gap Rescue 사용",
            variable=self.rescue_var,
        ).grid(row=1, column=2, sticky="w", pady=(12, 0))

        queue_frame = ttk.LabelFrame(self.root, text="작업 대상", padding=16)
        queue_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 12))
        queue_frame.columnconfigure(0, weight=1)
        queue_frame.rowconfigure(1, weight=1)

        drop_text = "여기에 파일을 드래그앤드롭하거나 아래 버튼으로 추가"
        if not TkinterDnD:
            drop_text += "  |  드래그앤드롭을 쓰려면 `pip install tkinterdnd2`"
        self.drop_label = ttk.Label(queue_frame, text=drop_text, anchor="center", relief="groove", padding=20)
        self.drop_label.grid(row=0, column=0, sticky="ew")
        if TkinterDnD and DND_FILES:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop)

        self.task_list = Listbox(queue_frame, height=10)
        self.task_list.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        actions = ttk.Frame(queue_frame)
        actions.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(5, weight=1)

        self.add_files_button = ttk.Button(actions, text="파일 추가", command=self._add_files)
        self.add_files_button.grid(row=0, column=0, padx=(0, 8))

        self.add_folder_button = ttk.Button(actions, text="폴더 추가", command=self._add_folder)
        self.add_folder_button.grid(row=0, column=1, padx=(0, 8))

        ttk.Button(actions, text="기본 입력 불러오기", command=self._load_default_targets).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(actions, text="선택 제거", command=self._remove_selected).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(actions, text="전체 비우기", command=self._clear_tasks).grid(row=0, column=4)

        run_frame = ttk.Frame(self.root, padding=(16, 0, 16, 12))
        run_frame.grid(row=3, column=0, sticky="nsew")
        run_frame.columnconfigure(0, weight=1)
        run_frame.rowconfigure(1, weight=1)

        controls = ttk.Frame(run_frame)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(controls, text="실행", command=self._start_processing)
        self.run_button.grid(row=0, column=0)
        ttk.Label(controls, textvariable=self.status_var).grid(row=0, column=1, sticky="w", padx=(12, 0))

        self.log_text = Text(run_frame, wrap="word", state="disabled")
        self.log_text.grid(row=1, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(run_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _sync_translate_mode(self) -> None:
        translate_only = self.translate_only_var.get()
        if translate_only:
            self.skip_translate_var.set(False)
            self.rescue_var.set(False)
            self.mode_var.set("movie")
        self._refresh_drop_state()

    def _refresh_drop_state(self) -> None:
        if self.translate_only_var.get():
            self.drop_label.configure(text="JSON 세그먼트 파일 또는 폴더를 추가하세요.")
        elif TkinterDnD:
            self.drop_label.configure(text="영상 파일 또는 폴더를 드래그앤드롭하거나 아래 버튼으로 추가")
        else:
            self.drop_label.configure(
                text="영상 파일 또는 폴더를 추가하세요. 드래그앤드롭을 쓰려면 `pip install tkinterdnd2`"
            )

    def _append_log(self, text: str) -> None:
        self.log_queue.put(text)

    def _flush_logs(self) -> None:
        while not self.log_queue.empty():
            chunk = self.log_queue.get_nowait()
            self.log_text.configure(state="normal")
            self.log_text.insert("end", chunk)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.root.after(150, self._flush_logs)

    def _add_paths(self, paths: list[Path]) -> None:
        accepted = set()
        translate_only = self.translate_only_var.get()
        for path in paths:
            if not path.exists():
                continue
            if path.is_dir():
                candidates = get_json_files(path) if translate_only else get_video_files(path)
                for candidate in candidates:
                    accepted.add(candidate.resolve())
                continue

            suffix = path.suffix.lower()
            if translate_only and suffix == ".json":
                accepted.add(path.resolve())
            elif not translate_only and suffix in VIDEO_EXTENSIONS:
                accepted.add(path.resolve())

        for path in sorted(accepted):
            if path not in self.task_paths:
                self.task_paths.append(path)
                self.task_list.insert("end", str(path))

        self.status_var.set(f"{len(self.task_paths)}개 대상 준비")

    def _on_drop(self, event) -> None:
        self._add_paths(split_drop_paths(event.data))

    def _add_files(self) -> None:
        if self.translate_only_var.get():
            files = filedialog.askopenfilenames(filetypes=[("JSON files", "*.json")])
        else:
            files = filedialog.askopenfilenames(
                filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.flv *.webm *.wmv")]
            )
        self._add_paths([Path(file) for file in files])

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self._add_paths([Path(folder)])

    def _load_default_targets(self) -> None:
        if self.translate_only_var.get():
            targets = get_json_files(Path(config.OUTPUT_DIR))
        else:
            targets = collect_tasks(None, False)
        self._add_paths(targets)

    def _remove_selected(self) -> None:
        selected = list(self.task_list.curselection())
        for index in reversed(selected):
            self.task_list.delete(index)
            del self.task_paths[index]
        self.status_var.set(f"{len(self.task_paths)}개 대상 준비")

    def _clear_tasks(self) -> None:
        self.task_paths.clear()
        self.task_list.delete(0, "end")
        self.status_var.set("대기 중")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.add_files_button.configure(state=state)
        self.add_folder_button.configure(state=state)

    def _start_processing(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if not self.task_paths:
            messagebox.showwarning("대상 없음", "처리할 파일이나 폴더를 먼저 추가하세요.")
            return

        options = ProcessingOptions(
            model_name=self.model_var.get().strip() or config.MODEL_NAME,
            mode=self.mode_var.get(),
            skip_translate=self.skip_translate_var.get(),
            translate_only=self.translate_only_var.get(),
            enable_rescue=self.rescue_var.get(),
        )

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self._set_running(True)
        self.status_var.set("실행 중")

        def worker() -> None:
            try:
                run_batch(self.task_paths, options, log_callback=self._append_log)
                self.log_queue.put("\n[GUI] 작업이 완료되었습니다.\n")
                self.root.after(0, lambda: self.status_var.set("완료"))
            except Exception as exc:
                self.log_queue.put(f"\n[GUI][ERROR] {exc}\n")
                self.root.after(0, lambda: self.status_var.set("오류"))
            finally:
                self.root.after(0, lambda: self._set_running(False))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    SubtitleGui().run()


if __name__ == "__main__":
    main()
