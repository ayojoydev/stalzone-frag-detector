"""Color console menus with immediate digit/arrow input (no shell commands)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


RESET = "\033[0m"
CYAN = "\033[96m"
DIM = "\033[90m"
WHITE = "\033[97m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
SELECTED = "\033[48;5;24m\033[97m"

# Values are deliberately selectable, rather than requiring flag/value typing.
PRESETS = (
    ("cut_before", "Секунд до фрага", (0, 3, 5, 10, 15, 20, 30, 45, 60)),
    ("cut_after", "Секунд после фрага", (0, 3, 5, 10, 15, 20, 30, 45, 60)),
    ("multifrag_gap", "Интервал мультифрага, сек.", (0, 2, 3, 5, 7, 10, 15, 20, 30)),
    ("frame_step", "Проверять каждый N-й кадр", (1, 5, 10, 15, 20, 30, 60, 90, 120)),
    ("threshold", "Порог совпадения", (0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.82, 0.9)),
    ("min_hits", "Подтверждений на событие", (1, 2, 3, 4, 5, 6, 8, 10, 15)),
    ("merge_gap", "Склейка кадров одной плашки, сек.", (0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3)),
    ("death_rescan", "Проверка перед смертью, сек.", (0, 1, 2, 3, 4, 5, 7, 10, 15)),
    ("death_frame_step", "Шаг проверки перед смертью", (1, 2, 3, 5, 10, 15, 20, 30, 60)),
    ("death_threshold", "Порог экрана смерти", (0.55, 0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.82, 0.9)),
    ("plaque_duration", "Время иконки одного килла, сек.", (1, 2, 3, 4, 5, 6, 7, 8, 10)),
)


def enable_color() -> bool:
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        import ctypes

        kernel = ctypes.windll.kernel32
        handle = kernel.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel.SetConsoleMode(handle, mode.value | 0x0004))
    return True


def read_key() -> str:
    if sys.platform == "win32":
        import msvcrt

        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(msvcrt.getwch(), "")
        return {"\r": "enter", "\x1b": "escape", "\x03": "escape"}.get(key, key)
    import termios
    import tty
    import select

    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = os.read(fd, 1).decode("utf-8", errors="replace")
        if key == "\x1b" and select.select([fd], [], [], 0.05)[0]:
            suffix = os.read(fd, 2).decode("ascii", errors="ignore")
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}.get(suffix, "")
        return {"\r": "enter", "\n": "enter", "\x1b": "escape", "\x03": "escape"}.get(key, key)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def navigate(selected: int, count: int, key: str) -> tuple[int, bool]:
    """Return selection and whether it was activated; shared by all menu pages."""
    if key in ("up", "left"):
        return (selected - 1) % count, False
    if key in ("down", "right", "\t"):
        return (selected + 1) % count, False
    if key in "123456789" and len(key) == 1 and int(key) <= count:
        return int(key) - 1, True
    return selected, key == "enter"


class TerminalApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.input = args.input_option or args.input_path
        self.output = args.output
        self.cut = True
        self.debug = args.debug
        self.exact = args.cut_mode == "exact"
        self.color = enable_color()
        self.result_dir: Path | None = None
        self.notice = "Выберите запись. Исходное видео не изменяется."

    def paint(self, text: str, color: str) -> str:
        return color + text + RESET if self.color else text

    def clear(self) -> None:
        if self.color:
            print("\033[2J\033[H", end="", flush=True)
        elif sys.platform == "win32":
            os.system("cls")
        else:
            print("\n" * 30)

    def draw(self, title: str, labels: list[str], selected: int, detail: str = "") -> None:
        from .cli import VERSION

        width = min(96, max(35, shutil.get_terminal_size((90, 30)).columns - 2))
        self.clear()
        print(self.paint("━" * width, CYAN))
        print(self.paint(f"  SZFD {VERSION}  /  ДЕТЕКТОР ФРАГОВ", CYAN))
        print(self.paint("  АВТОМАСШТАБ · 4:3 / 16:9 / ULTRAWIDE / 4K", DIM))
        print(self.paint("━" * width, CYAN))
        print("\n  " + self.paint(title, WHITE) + "\n")
        for index, label in enumerate(labels):
            line = f" {'›' if selected == index else ' '} {index + 1}  {label}"
            if len(line) > width:
                line = line[:width - 1] + "…"
            print(self.paint(line.ljust(width), SELECTED if selected == index else WHITE))
        print("\n" + self.paint("─" * width, DIM))
        if detail:
            import textwrap
            for line in textwrap.wrap(detail, max(30, width - 4)):
                print(self.paint("  " + line, YELLOW))
        print(self.paint("  1–9 выбор   ↑↓←→ навигация   Enter открыть   Esc назад", CYAN))
        sys.stdout.flush()

    def menu(self, title: str, labels: list[str], selected: int = 0, detail: str = "") -> int | None:
        while True:
            self.draw(title, labels, selected, detail)
            key = read_key()
            if key == "escape":
                return None
            selected, activated = navigate(selected, len(labels), key)
            if activated:
                return selected

    def pick_path(self, kind: str) -> Path | None:
        # Only the OS file picker is graphical; the application itself stays in the console.
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                if kind == "output":
                    path = filedialog.askdirectory(parent=root, title="Папка результатов")
                else:
                    types = [("Все файлы", "*.*")]
                    if kind == "input":
                        types.insert(0, ("Видео / скриншоты", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.ts *.png *.jpg *.jpeg *.bmp *.webp"))
                    elif kind in ("template", "death_template"):
                        types.insert(0, ("Шаблоны", "*.png *.jpg *.bmp"))
                    path = filedialog.askopenfilename(parent=root, title="Выберите файл", filetypes=types)
            finally:
                root.destroy()
            return Path(path) if path else None
        except Exception as exc:
            self.notice = f"Не удалось открыть выбор файла: {exc}"
            return None

    def choose_preset(self, name: str, label: str, choices: tuple) -> None:
        current = getattr(self.args, name)
        selected = min(range(len(choices)), key=lambda i: abs(choices[i] - current))
        detail = f"Текущее значение: {current}. Esc — оставить без изменений."
        if name == "merge_gap":
            detail += (" Это интервал между кадрами ОДНОЙ иконки, а не между убийствами. "
                       "Он не может быть меньше шага проверки кадров: программа поднимет "
                       "его автоматически. Для серии убийств меняйте интервал мультифрага.")
        index = self.menu(label, [str(value) for value in choices], selected, detail)
        if index is not None:
            setattr(self.args, name, choices[index])

    def settings(self) -> None:
        page = 0
        selected = 0
        while True:
            # Seven settings plus next/back: every page fits in the 1–9 keys.
            options = [("multikill", "Мультикиллы · целая серия", None)] + list(PRESETS) + [
                ("roi", "Область поиска фрага", None),
                ("death_roi", "Область экрана смерти", None),
                ("template", "Шаблон фрага", None),
                ("death_template", "Шаблон экрана смерти", None),
                ("ffmpeg", "FFmpeg", None),
            ]
            pages = (len(options) + 6) // 7
            items = options[page * 7:(page + 1) * 7]
            labels = [f"{label}: " + (
                ("ВКЛ" if self.args.multikill else "ВЫКЛ") if name == "multikill"
                else str(getattr(self.args, name)) if getattr(self.args, name) is not None else "автоматически"
            ) for name, label, _ in items]
            labels += ["Следующая страница →", "← Главное меню"]
            index = self.menu(f"НАСТРОЙКИ · страница {page + 1}/{pages}", labels, min(selected, len(labels) - 1))
            if index is None or index == len(items) + 1:
                return
            selected = index
            if index == len(items):
                page = (page + 1) % pages
                selected = 0
                continue
            name, label, choices = items[index]
            if name == "multikill":
                self.args.multikill = not self.args.multikill
                if self.args.multikill:
                    self.cut = True
            elif choices:
                self.choose_preset(name, label, choices)
            elif name in ("roi", "death_roi"):
                from .cli import DEFAULT_ROI, DEFAULT_DEATH_ROI

                default = DEFAULT_ROI if name == "roi" else DEFAULT_DEATH_ROI
                regions = [default, (0.2, 0.5, 0.8, 0.98) if name == "roi" else (0, 0, 0.6, 0.5), (0, 0, 1, 1)]
                choice = self.menu(label, ["Стандартная (адаптация пропорций)", "Расширенная область", "Весь кадр (медленнее)"])
                if choice is not None:
                    setattr(self.args, name, regions[choice])
            else:
                choice = self.menu(label, ["Выбрать файл", "Вернуть автоматический / стандартный путь"])
                if choice == 0:
                    path = self.pick_path(name)
                    if path:
                        setattr(self.args, name, path)
                elif choice == 1:
                    from .cli import build_parser

                    setattr(self.args, name, getattr(build_parser().parse_args([]), name))

    def processing_arguments(self) -> list[str]:
        values = {name: getattr(self.args, name) for name, _, _ in PRESETS}
        values.update({name: getattr(self.args, name) for name in ("roi", "death_roi", "template", "death_template", "ffmpeg")})
        result = ["--input", str(self.input)]
        if self.output:
            result += ["--output", str(self.output)]
        for name, value in values.items():
            if value is not None:
                result += ["--" + name.replace("_", "-"), ",".join(map(str, value)) if isinstance(value, tuple) else str(value)]
        if self.cut:
            result.append("--cut")
        if self.debug:
            result.append("--debug")
        if self.args.multikill:
            result.append("--multikill")
        result += ["--cut-mode", "exact" if self.exact else "copy"]
        return result

    def start(self) -> None:
        from .cli import build_parser, main, resolve_output_paths, validate_args

        if self.input is None:
            self.notice = "Сначала выберите видео: пункт 1."
            return
        arguments = self.processing_arguments()
        parser = build_parser()
        def error(message: str) -> None:
            raise ValueError(message)
        parser.error = error
        try:
            args = parser.parse_args(arguments)
            validate_args(args, parser)
            for name in ("template", "death_template"):
                if not getattr(args, name).is_file():
                    raise ValueError(f"Не найден шаблон: {getattr(args, name)}")
        except ValueError as exc:
            self.notice = f"Проверьте настройки: {exc}"
            return
        self.clear()
        print(self.paint("\n  ▶ ОБРАБОТКА ЗАПИСИ", CYAN))
        print(self.paint("  Ctrl+C — остановить. Исходное видео не изменяется.\n", DIM))
        try:
            code = main(arguments)
            csv, artifacts = resolve_output_paths(self.input, self.output, self.cut)
            self.result_dir = artifacts if self.cut else csv.parent
            self.notice = {0: "Готово! Пункт 8 — открыть результаты.",
                           2: "Плашка не найдена. Попробуйте изменить настройки."}.get(code, "Ошибка обработки. Проверьте сообщение выше.")
        except KeyboardInterrupt:
            self.notice = "Обработка остановлена. Уже сохранённые результаты остались на диске."
        except Exception as exc:
            self.notice = f"Ошибка: {exc}"
        print("\n" + self.paint("  " + self.notice, GREEN))
        print(self.paint("  Нажмите любую клавишу, чтобы вернуться в меню…", CYAN))
        read_key()

    def open_results(self) -> None:
        if not self.result_dir or not self.result_dir.is_dir():
            self.notice = "Результатов ещё нет. Сначала запустите обработку."
            return
        try:
            if sys.platform == "win32":
                os.startfile(self.result_dir)
            else:
                import subprocess

                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(self.result_dir)])
        except OSError as exc:
            self.notice = f"Не удалось открыть папку: {exc}"

    def run(self) -> int:
        selected = 0
        while True:
            labels = [f"Запись: {self.input or 'выбрать видео / скриншот'}",
                      f"Результаты: {self.output or 'рядом с видео'}",
                      f"Нарезка клипов: {'ВКЛ · МУЛЬТИКИЛЛЫ' if self.args.multikill else 'ВКЛ' if self.cut else 'ВЫКЛ'}",
                      f"Проверочные кадры: {'ВКЛ' if self.debug else 'ВЫКЛ'}",
                      f"Режим: {'ТОЧНЫЙ · с перекодированием' if self.exact else 'БЫСТРЫЙ · без пережатия'}",
                      "Настройки распознавания и нарезки", "▶ НАЧАТЬ ОБРАБОТКУ", "Открыть результаты", "Выход"]
            index = self.menu("ГЛАВНОЕ МЕНЮ", labels, selected, self.notice)
            if index is None or index == 8:
                self.clear()
                print(self.paint("  SZFD · До встречи!", CYAN))
                return 0
            selected = index
            if index in (0, 1):
                path = self.pick_path("input" if index == 0 else "output")
                if path:
                    if index == 0:
                        self.input = path
                    else:
                        self.output = path
            elif index == 2:
                self.cut = not self.cut
                if not self.cut:
                    self.args.multikill = False
            elif index == 3:
                self.debug = not self.debug
            elif index == 4:
                self.exact = not self.exact
            elif index == 5:
                self.settings()
            elif index == 6:
                self.start()
            elif index == 7:
                self.open_results()


def launch(args: argparse.Namespace) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("Интерактивному меню нужна консоль. Откройте SZFD.exe двойным щелчком "
              "или используйте --input PATH для автоматического запуска.", file=sys.stderr)
        return 1
    try:
        return TerminalApplication(args).run()
    finally:
        if sys.stdout.isatty():
            print(RESET, end="", flush=True)
