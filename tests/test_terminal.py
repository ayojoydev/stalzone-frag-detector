import pytest

from szfd.cli import build_parser, validate_args
from szfd.terminal import PRESETS, TerminalApplication, navigate


@pytest.mark.parametrize("key,selected,expected", [
    ("up", 0, (8, False)), ("left", 0, (8, False)),
    ("right", 8, (0, False)), ("down", 8, (0, False)),
    ("7", 0, (6, True)), ("enter", 4, (4, True)),
    ("", 0, (0, False)), ("x", 3, (3, False)),
])
def test_navigation(key, selected, expected):
    assert navigate(selected, 9, key) == expected


def test_digit_outside_submenu_is_ignored():
    assert navigate(0, 3, "9") == (0, False)


def test_processing_arguments(tmp_path):
    video = tmp_path / "запись с пробелами.mp4"
    video.touch()
    args = build_parser().parse_args(["--menu", "--input", str(video), "--death-rescan", "0"])
    app = TerminalApplication(args)
    app.exact = True
    app.debug = True
    app.args.cut_before = 15
    arguments = app.processing_arguments()
    assert "--menu" not in arguments
    assert "--gui" not in arguments
    parser = build_parser()
    result = parser.parse_args(arguments)
    validate_args(result, parser)
    assert result.input == video
    assert result.death_rescan == 0
    assert result.cut_before == 15
    assert result.cut and result.debug and result.cut_mode == "exact"


def test_multikill_menu_option_reaches_shared_cli(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.touch()
    app = TerminalApplication(build_parser().parse_args(["--input", str(video)]))
    app.cut = False
    choices = iter([0, 8])
    monkeypatch.setattr(app, "menu", lambda *args, **kwargs: next(choices))
    app.settings()
    assert app.cut and app.args.multikill
    parser = build_parser()
    args = parser.parse_args(app.processing_arguments())
    validate_args(args, parser)
    assert args.multikill and args.cut


def test_presets_are_valid(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    parser = build_parser()
    for name, _, choices in PRESETS:
        assert len(choices) <= 9
        for value in choices:
            args = parser.parse_args(["--input", str(video), "--" + name.replace("_", "-") + "=" + str(value)])
            validate_args(args, parser)


def test_menu_selects_digit_without_enter(monkeypatch):
    app = TerminalApplication(build_parser().parse_args([]))
    monkeypatch.setattr(app, "draw", lambda *args: None)
    monkeypatch.setattr("szfd.terminal.read_key", lambda: "3")
    assert app.menu("test", ["a", "b", "c"]) == 2


def test_menu_escape(monkeypatch):
    app = TerminalApplication(build_parser().parse_args([]))
    monkeypatch.setattr(app, "draw", lambda *args: None)
    monkeypatch.setattr("szfd.terminal.read_key", lambda: "escape")
    assert app.menu("test", ["a"]) is None


def test_settings_return_without_typing(monkeypatch):
    app = TerminalApplication(build_parser().parse_args([]))
    # First setting -> 15 seconds -> last (back) entry on first page.
    selections = iter([1, 4, 8])
    monkeypatch.setattr(app, "menu", lambda *args, **kwargs: next(selections))
    app.settings()
    assert app.args.cut_before == 15


def test_processing_returns_to_menu_on_error(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.touch()
    app = TerminalApplication(build_parser().parse_args(["--input", str(video)]))
    monkeypatch.setattr(app, "clear", lambda: None)
    monkeypatch.setattr("szfd.cli.main", lambda args: 1)
    monkeypatch.setattr("szfd.terminal.read_key", lambda: "enter")
    app.start()
    assert "Ошибка" in app.notice
