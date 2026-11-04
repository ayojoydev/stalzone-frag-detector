"""Keyboard-friendly desktop launcher; processing runs in an isolated child process."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# All detector options stay available without typing command-line flags.
FIELDS = (
    ("cut_before", "Секунд до фрага"), ("cut_after", "Секунд после фрага"),
    ("multifrag_gap", "Интервал мультифрага (сек.)"),
    ("frame_step", "Проверять каждый N-й кадр"), ("threshold", "Порог совпадения (0–1)"),
    ("min_hits", "Подтверждений на событие"), ("merge_gap", "Объединение срабатываний (сек.)"),
    ("death_rescan", "Повторная проверка перед смертью (сек.)"),
    ("death_frame_step", "Шаг повторной проверки"),
    ("death_threshold", "Порог экрана смерти (0–1)"),
    ("roi", "Область фрага: x1,y1,x2,y2"),
    ("death_roi", "Область смерти: x1,y1,x2,y2"),
    ("template", "Шаблон значка (PNG)"), ("death_template", "Шаблон смерти (PNG)"),
    ("ffmpeg", "Путь к FFmpeg (пусто = автоматически)"),
    ("plaque_duration", "Время иконки одного килла (сек.)"),
)


def detector_arguments(values: dict[str, str], cut: bool, debug: bool, exact: bool) -> list[str]:
    result = []
    for name, value in values.items():
        if value.strip():
            result.extend(["--" + name.replace("_", "-"), value.strip()])
    if cut:
        result.append("--cut")
    if debug:
        result.append("--debug")
    result.extend(["--cut-mode", "exact" if exact else "copy"])
    return result


class Application:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        root.title("SZFD — поиск и нарезка фрагов")
        root.geometry("900x720")
        root.minsize(650, 580)
        self.values = {}
        for name, _ in FIELDS:
            value = getattr(args, name)
            self.values[name] = ",".join(map(str, value)) if isinstance(value, tuple) else ("" if value is None else str(value))
        self.input = str(args.input_option or args.input_path or "")
        self.output = str(args.output or "")
        self.cut, self.debug, self.exact = True, args.debug, args.cut_mode == "exact"
        self.multikill = args.multikill
        self.process: subprocess.Popen | None = None
        self.busy = False
        self.messages: queue.Queue = queue.Queue()
        self.selected = 0
        self.result_dir: Path | None = None
        panel = ttk.Frame(root, padding=16)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="SZFD · Детектор фрагов", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(panel, text="Цифры 1–9 — выбрать пункт · Стрелки — перемещение · Enter — выбрать").pack(anchor="w", pady=8)
        self.buttons = []
        actions = [self.choose_input, self.choose_output, self.toggle_cut, self.toggle_debug,
                   self.toggle_exact, self.settings, self.start, self.open_results, self.stop_or_exit]
        for index, action in enumerate(actions):
            button = ttk.Button(panel, command=action, takefocus=True)
            button.pack(fill="x", pady=2)
            self.buttons.append(button)
            button.bind("<FocusIn>", lambda event, i=index: setattr(self, "selected", i))
            root.bind(str(index + 1), lambda event, i=index: self.invoke(i))
        self.status = ttk.Label(panel, text="Выберите видео или изображение. Исходный файл не изменяется.", wraplength=800)
        self.status.pack(anchor="w", pady=8)
        self.log = tk.Text(panel, height=10, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        for key, step in (("Up", -1), ("Left", -1), ("Down", 1), ("Right", 1)):
            root.bind("<" + key + ">", lambda event, s=step: self.move(s))
        root.bind("<Return>", lambda event: self.invoke(self.selected))
        root.protocol("WM_DELETE_WINDOW", self.stop_or_exit)
        self.refresh()
        self.buttons[0].focus_set()
        root.after(100, self.poll)

    def refresh(self) -> None:
        labels = [f"Видео / скриншот: {self.input or 'не выбран'}",
                  f"Папка результатов: {self.output or 'рядом с видео'}",
                  f"Нарезка клипов: {'ВКЛ' if self.cut else 'ВЫКЛ'}",
                  f"Проверочные кадры: {'ВКЛ' if self.debug else 'ВЫКЛ'}",
                  f"Режим: {'точная нарезка с перекодированием' if self.exact else 'быстро, без пережатия'}",
                  "Дополнительные настройки", "Начать обработку", "Открыть результаты",
                  "Остановить обработку" if self.busy else "Выход"]
        for index, (button, label) in enumerate(zip(self.buttons, labels)):
            button.configure(text=f"{index + 1}. {label}", state="disabled" if self.busy and index < 7 else "normal")

    def invoke(self, index: int) -> str:
        self.selected = index
        self.buttons[index].focus_set()
        self.buttons[index].invoke()
        return "break"

    def move(self, step: int) -> str:
        self.selected = (self.selected + step) % 9
        self.buttons[self.selected].focus_set()
        return "break"

    def choose_input(self) -> None:
        path = filedialog.askopenfilename(title="Выберите видео или скриншот", filetypes=[
            ("Видео и изображения", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.ts *.png *.jpg *.jpeg *.bmp *.webp"),
            ("Все файлы", "*.*")])
        if path:
            self.input = path
            self.refresh()

    def choose_output(self) -> None:
        path = filedialog.askdirectory(title="Куда сохранить результаты?")
        if path:
            self.output = path
            self.refresh()

    def toggle_cut(self) -> None:
        self.cut = not self.cut
        if not self.cut:
            self.multikill = False
        self.refresh()

    def toggle_debug(self) -> None:
        self.debug = not self.debug
        self.refresh()

    def toggle_exact(self) -> None:
        self.exact = not self.exact
        self.refresh()

    def settings(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Настройки распознавания и нарезки")
        dialog.transient(self.root)
        dialog.grab_set()
        entries = {}
        multikill = tk.BooleanVar(value=self.multikill)
        for row, (name, label) in enumerate(FIELDS):
            ttk.Label(dialog, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=4)
            entry = ttk.Entry(dialog, width=45)
            entry.insert(0, self.values[name])
            entry.grid(row=row, column=1, padx=10, pady=4, sticky="ew")
            entries[name] = entry

        def save() -> None:
            values = {name: entry.get() for name, entry in entries.items()}
            previous = self.multikill
            self.multikill = multikill.get()
            if self.validate(values, require_input=False):
                self.values = values
                if self.multikill:
                    self.cut = True
                self.refresh()
                dialog.destroy()
            else:
                self.multikill = previous

        ttk.Checkbutton(dialog, text="Мультикиллы — сохранять серии целиком", variable=multikill).grid(row=len(FIELDS), columnspan=2, pady=4)
        ttk.Button(dialog, text="Сохранить", command=save).grid(row=len(FIELDS) + 1, column=1, pady=12)
        dialog.bind("<Escape>", lambda event: dialog.destroy())
        entries["cut_before"].focus_set()

    def validate(self, values: dict[str, str], require_input: bool = True) -> bool:
        from .cli import build_parser, validate_args
        # argparse errors are shown in a dialog instead of closing the app.
        parser = build_parser()
        def error(message: str) -> None:
            raise ValueError(message)
        parser.error = error
        try:
            arguments = detector_arguments(values, self.cut, self.debug, self.exact)
            if self.multikill:
                arguments.append("--multikill")
            args = parser.parse_args(["--input", self.input if require_input else __file__, *arguments])
            if self.output:
                args.output = Path(self.output)
            validate_args(args, parser)
            for name in ("template", "death_template"):
                if not getattr(args, name).is_file():
                    raise ValueError(f"Не найден шаблон: {getattr(args, name)}")
        except (ValueError, argparse.ArgumentError) as exc:
            messagebox.showerror("Проверьте настройки", str(exc), parent=self.root)
            return False
        return True

    def start(self) -> None:
        if self.busy or not self.validate(self.values):
            return
        from .cli import resolve_output_paths
        csv, artifacts = resolve_output_paths(Path(self.input), Path(self.output) if self.output else None, self.cut or self.multikill)
        self.result_dir = artifacts if self.cut or self.multikill else csv.parent
        command = ([sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "szfd"])
        command += ["--input", self.input, *detector_arguments(self.values, self.cut, self.debug, self.exact)]
        if self.multikill:
            command.append("--multikill")
        if self.output:
            command += ["--output", self.output]
        self.busy = True
        self.refresh()
        self.status.configure(text="Обработка… Результаты и прогресс появляются ниже.")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                     "PYINSTALLER_RESET_ENVIRONMENT": "1"},
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        except OSError as exc:
            self.busy = False
            self.refresh()
            messagebox.showerror("Не удалось запустить обработку", str(exc))
            return
        process = self.process
        def read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self.messages.put(line)
            self.messages.put(process.wait())
        threading.Thread(target=read_output, daemon=True).start()

    def poll(self) -> None:
        for _ in range(200):
            try:
                item = self.messages.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, int):
                self.busy = False
                self.process = None
                self.refresh()
                self.status.configure(text={0: "Готово! Нажмите 8, чтобы открыть результаты.",
                    2: "Плашка не найдена. Попробуйте изменить настройки."}.get(item, "Обработка прервана или произошла ошибка. Подробности ниже."))
            else:
                self.log.configure(state="normal")
                self.log.insert("end", item)
                if int(self.log.index("end-1c").split(".")[0]) > 1000:
                    self.log.delete("1.0", "200.0")
                self.log.see("end")
                self.log.configure(state="disabled")
        self.root.after(100, self.poll)

    def open_results(self) -> None:
        if self.result_dir and self.result_dir.is_dir():
            if sys.platform == "win32":
                os.startfile(self.result_dir)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(self.result_dir)])
        else:
            messagebox.showinfo("Результаты", "Сначала запустите обработку.")

    def stop_or_exit(self) -> None:
        if self.busy:
            if messagebox.askyesno("Остановить?", "Прервать обработку? Уже сохранённые результаты останутся."):
                # Include FFmpeg, if it is currently cutting a clip.
                if self.process and self.process.poll() is None:
                    if sys.platform == "win32":
                        subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                                       creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
                    else:
                        self.process.terminate()
            return
        self.root.destroy()


def launch(args: argparse.Namespace) -> int:
    if sys.platform == "win32" and getattr(sys, "frozen", False):
        import ctypes
        console = ctypes.windll.kernel32.GetConsoleWindow()
        if console:
            ctypes.windll.user32.ShowWindow(console, 0)
    root = tk.Tk()
    Application(root, args)
    root.mainloop()
    return 0
