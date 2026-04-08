from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR_NAME = SCRIPT_DIR.name
DEFAULT_WINDOWS_INSTALL_DIR = Path(r"C:\Program Files\MetaTrader 5")
DEFAULT_WINDOWS_TERMINAL_PATH = DEFAULT_WINDOWS_INSTALL_DIR / "terminal64.exe"
DEFAULT_WINDOWS_METAEDITOR_PATH = DEFAULT_WINDOWS_INSTALL_DIR / "MetaEditor64.exe"
TERMINAL_EXECUTABLE_NAMES = ("terminal64.exe", "Terminal64.exe")
METAEDITOR_EXECUTABLE_NAMES = ("metaeditor64.exe", "MetaEditor64.exe")
METATESTER_EXECUTABLE_NAMES = ("metatester64.exe", "MetaTester64.exe")


@dataclass(frozen=True)
class Mt5RuntimePaths:
    host_platform: str
    use_wine: bool
    wineprefix: Path | None
    instance_root: Path
    terminal_path: Path
    metaeditor_path: Path
    expert_dir: Path
    files_dir: Path
    presets_dir: Path
    tester_profile_dir: Path
    tester_dir: Path
    terminal_log_dir: Path
    portable_mode: bool

    @property
    def deployed_live_mq5(self) -> Path:
        return self.expert_dir / "live.mq5"

    @property
    def deployed_live_ex5(self) -> Path:
        return self.expert_dir / "live.ex5"

    @property
    def deployed_compile_log(self) -> Path:
        return self.expert_dir / "live.compile.log"

    @property
    def deployed_model_path(self) -> Path:
        return self.expert_dir / "model.onnx"

    @property
    def deployed_model_config_path(self) -> Path:
        return self.expert_dir / "model_config.mqh"

    @property
    def deployed_shared_config_path(self) -> Path:
        return self.expert_dir / "shared_config.mqh"

    @property
    def expert_resource_name(self) -> str:
        return f"{PROJECT_DIR_NAME}\\live.ex5"


def host_platform_name() -> str:
    system_name = platform.system().lower()
    if system_name.startswith("win"):
        return "windows"
    if system_name == "linux":
        return "linux"
    if system_name == "darwin":
        raise EnvironmentError("MetaTrader automation is not supported on macOS. Use Windows or Linux with Wine.")
    raise EnvironmentError(f"Unsupported operating system: {platform.system()}")


def _candidate_wineprefixes() -> list[Path]:
    candidates: list[Path] = []
    env_wineprefix = os.environ.get("WINEPREFIX", "").strip()
    if env_wineprefix:
        _append_unique(candidates, Path(env_wineprefix))

    for candidate in SCRIPT_DIR.parents:
        if candidate.name.startswith("drive_") and candidate.parent.name in {".wine", ".mt5"}:
            _append_unique(candidates, candidate.parent)

    _append_unique(candidates, Path.home() / ".wine")
    _append_unique(candidates, Path.home() / ".mt5")
    return [candidate for candidate in candidates if candidate.exists()]


def default_linux_install_dirs() -> list[Path]:
    install_dirs: list[Path] = []
    for wineprefix in _candidate_wineprefixes():
        _append_unique(install_dirs, wineprefix / "drive_c" / "Program Files" / "MetaTrader 5")
    return install_dirs


def default_linux_wineprefix() -> Path | None:
    candidates = _candidate_wineprefixes()
    return candidates[0] if candidates else None


def is_instance_root(path: Path) -> bool:
    return (path / "MQL5").is_dir() and (path / "Tester").is_dir()


def find_instance_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if is_instance_root(candidate):
            return candidate
    return None


def read_text_best_effort(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-16", "utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _append_unique(paths: list[Path], candidate: Path | None) -> None:
    if candidate is None:
        return
    normalized = candidate.expanduser().resolve()
    if normalized not in paths:
        paths.append(normalized)


def _existing_candidates(directory: Path, names: tuple[str, ...]) -> list[Path]:
    return [directory / name for name in names if (directory / name).exists()]


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path.expanduser().resolve()
    return None


def _path_score(candidate: Path) -> tuple[int, float]:
    score = 0
    if (candidate / "MQL5" / "Experts" / PROJECT_DIR_NAME).exists():
        score += 100
    if _first_existing(_existing_candidates(candidate, TERMINAL_EXECUTABLE_NAMES)):
        score += 50
    if (candidate / "origin.txt").exists():
        score += 10

    mtime = 0.0
    for probe in (candidate / "origin.txt", candidate / "MQL5", candidate / "Tester"):
        if probe.exists():
            mtime = max(mtime, probe.stat().st_mtime)
    return score, mtime


def _candidate_instance_roots(instance_root_override: str) -> list[Path]:
    candidates: list[Path] = []
    if instance_root_override:
        _append_unique(candidates, Path(instance_root_override))
        return candidates

    for env_name in ("MT5_INSTANCE_ROOT", "MQL5_DATA_FOLDER_PATH"):
        value = os.environ.get(env_name, "").strip()
        if value:
            _append_unique(candidates, Path(value))

    detected = find_instance_root(SCRIPT_DIR)
    _append_unique(candidates, detected)

    platform_name = host_platform_name()
    if platform_name == "linux":
        for install_dir in default_linux_install_dirs():
            _append_unique(candidates, install_dir)
    else:
        appdata_value = os.environ.get("APPDATA", "").strip()
        if appdata_value:
            terminal_data_root = Path(appdata_value).expanduser() / "MetaQuotes" / "Terminal"
            if terminal_data_root.exists():
                for child in terminal_data_root.iterdir():
                    if child.is_dir() and is_instance_root(child):
                        _append_unique(candidates, child)
        _append_unique(candidates, DEFAULT_WINDOWS_INSTALL_DIR)

    return [candidate for candidate in candidates if candidate.exists() and is_instance_root(candidate)]


def resolve_instance_root(instance_root_override: str = "") -> Path:
    candidates = _candidate_instance_roots(instance_root_override)
    if not candidates:
        raise FileNotFoundError(
            "Could not locate a MetaTrader 5 data folder. Pass --instance-root or set MT5_INSTANCE_ROOT/"
            "MQL5_DATA_FOLDER_PATH."
        )
    return max(candidates, key=_path_score)


def _resolve_explicit_existing_path(path_str: str) -> Path | None:
    value = path_str.strip()
    if not value:
        return None

    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    which_path = shutil.which(value)
    if which_path:
        return Path(which_path).resolve()

    raise FileNotFoundError(f"Path not found: {path_str}")


def resolve_terminal_path(instance_root: Path, terminal_path_override: str = "") -> Path:
    explicit = _resolve_explicit_existing_path(terminal_path_override or os.environ.get("MT5_TERMINAL_PATH", ""))
    if explicit is not None:
        return explicit

    direct = _first_existing(_existing_candidates(instance_root, TERMINAL_EXECUTABLE_NAMES))
    if direct is not None:
        return direct

    origin_path = instance_root / "origin.txt"
    if origin_path.exists():
        install_dir = Path(read_text_best_effort(origin_path).strip()).expanduser()
        install_path = _first_existing(_existing_candidates(install_dir, TERMINAL_EXECUTABLE_NAMES))
        if install_path is not None:
            return install_path

    platform_name = host_platform_name()
    fallback_dirs = default_linux_install_dirs() if platform_name == "linux" else [DEFAULT_WINDOWS_INSTALL_DIR]
    for fallback_dir in fallback_dirs:
        fallback = _first_existing(_existing_candidates(fallback_dir, TERMINAL_EXECUTABLE_NAMES))
        if fallback is not None:
            return fallback

    raise FileNotFoundError(
        "Could not locate terminal64.exe. Pass --terminal-path or set MT5_TERMINAL_PATH."
    )


def resolve_metaeditor_path(instance_root: Path, terminal_path: Path, metaeditor_path_override: str = "") -> Path:
    explicit = _resolve_explicit_existing_path(metaeditor_path_override or os.environ.get("MQL5_COMPILER_PATH", ""))
    if explicit is not None:
        return explicit

    for directory in (terminal_path.parent, instance_root):
        candidate = _first_existing(_existing_candidates(directory, METAEDITOR_EXECUTABLE_NAMES))
        if candidate is not None:
            return candidate

    platform_name = host_platform_name()
    fallback_dirs = default_linux_install_dirs() if platform_name == "linux" else [DEFAULT_WINDOWS_INSTALL_DIR]
    for fallback_dir in fallback_dirs:
        fallback = _first_existing(_existing_candidates(fallback_dir, METAEDITOR_EXECUTABLE_NAMES))
        if fallback is not None:
            return fallback

    raise FileNotFoundError(
        "Could not locate MetaEditor. Pass --metaeditor-path or set MQL5_COMPILER_PATH."
    )


def resolve_mt5_runtime(
    instance_root_override: str = "",
    terminal_path_override: str = "",
    metaeditor_path_override: str = "",
) -> Mt5RuntimePaths:
    platform_name = host_platform_name()
    instance_root = resolve_instance_root(instance_root_override)
    terminal_path = resolve_terminal_path(instance_root, terminal_path_override)
    metaeditor_path = resolve_metaeditor_path(instance_root, terminal_path, metaeditor_path_override)
    wineprefix = default_linux_wineprefix() if platform_name == "linux" else None

    return Mt5RuntimePaths(
        host_platform=platform_name,
        use_wine=(platform_name == "linux"),
        wineprefix=wineprefix,
        instance_root=instance_root,
        terminal_path=terminal_path,
        metaeditor_path=metaeditor_path,
        expert_dir=instance_root / "MQL5" / "Experts" / PROJECT_DIR_NAME,
        files_dir=instance_root / "MQL5" / "Files",
        presets_dir=instance_root / "MQL5" / "Presets",
        tester_profile_dir=instance_root / "MQL5" / "Profiles" / "Tester",
        tester_dir=instance_root / "Tester",
        terminal_log_dir=instance_root / "Tester" / "logs",
        portable_mode=(terminal_path.parent.resolve() == instance_root.resolve()),
    )


def ensure_runtime_dirs(runtime: Mt5RuntimePaths) -> None:
    for path in (
        runtime.expert_dir,
        runtime.files_dir,
        runtime.presets_dir,
        runtime.tester_profile_dir,
        runtime.tester_dir,
        runtime.terminal_log_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def iter_agent_log_paths(runtime: Mt5RuntimePaths) -> list[Path]:
    if not runtime.tester_dir.exists():
        return []
    return sorted(runtime.tester_dir.glob("Agent-*/logs/*.log"))


def _manual_wine_path(path: Path, wineprefix: Path) -> str:
    normalized = path.expanduser().resolve(strict=False)
    try:
        relative = normalized.relative_to(wineprefix)
        drive_root = relative.parts[0]
        if drive_root.startswith("drive_") and len(drive_root) == 7:
            drive_letter = drive_root[-1].upper()
            remainder = relative.parts[1:]
            if not remainder:
                return f"{drive_letter}:\\"
            return f"{drive_letter}:\\" + "\\".join(remainder)
    except ValueError:
        pass

    drive_path = normalized.as_posix().lstrip("/").replace("/", "\\")
    return f"Z:\\{drive_path}"


def to_windows_path(runtime: Mt5RuntimePaths, path: Path) -> str:
    if not runtime.use_wine:
        return str(path)

    if shutil.which("winepath"):
        completed = subprocess.run(
            ["winepath", "-w", str(path)],
            capture_output=True,
            text=True,
            check=False,
            env=runtime_env(runtime),
        )
        converted = completed.stdout.strip()
        if completed.returncode == 0 and converted:
            return converted

    if runtime.wineprefix is None:
        raise EnvironmentError("Wine prefix is required for Linux MetaTrader runtime handling.")
    return _manual_wine_path(path, runtime.wineprefix)


def runtime_env(runtime: Mt5RuntimePaths) -> dict[str, str]:
    env = os.environ.copy()
    if runtime.use_wine and runtime.wineprefix is not None:
        env["WINEPREFIX"] = str(runtime.wineprefix)
    return env


def build_terminal_command(runtime: Mt5RuntimePaths, config_path: Path) -> list[str]:
    config_value = to_windows_path(runtime, config_path) if runtime.use_wine else str(config_path)
    config_arg = f"/config:{config_value}"
    command = [str(runtime.terminal_path), config_arg]
    if runtime.portable_mode:
        command.append("/portable")
    if runtime.use_wine:
        return ["wine", *command]
    return command


def build_metaeditor_compile_command(
    runtime: Mt5RuntimePaths,
    source_path: Path,
    log_path: Path,
) -> list[str]:
    source_value = to_windows_path(runtime, source_path) if runtime.use_wine else str(source_path)
    log_value = to_windows_path(runtime, log_path) if runtime.use_wine else str(log_path)
    if runtime.use_wine:
        return [
            "wine",
            str(runtime.metaeditor_path),
            f'/compile:"{source_value}"',
            f'/log:"{log_value}"',
        ]
    return [
        str(runtime.metaeditor_path),
        f'/compile:"{source_value}"',
        f'/log:"{log_value}"',
    ]


def stop_terminal_best_effort(runtime: Mt5RuntimePaths) -> None:
    commands: list[list[str]]
    if runtime.use_wine:
        commands = [
            ["wine", "cmd", "/c", "taskkill", "/IM", "terminal64.exe", "/F"],
            ["pkill", "-f", "terminal64.exe"],
        ]
    else:
        commands = [["taskkill", "/IM", "terminal64.exe", "/F"]]

    for command in commands:
        try:
            subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=runtime_env(runtime),
            )
        except FileNotFoundError:
            continue


def launch_terminal(
    runtime: Mt5RuntimePaths,
    config_path: Path,
    timeout_seconds: int | None = None,
    detach: bool = False,
    stop_existing: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.Popen[str]:
    if stop_existing:
        stop_terminal_best_effort(runtime)

    command = build_terminal_command(runtime, config_path)
    env = runtime_env(runtime)
    if detach:
        popen_kwargs: dict[str, object] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": env,
            "text": True,
        }
        if runtime.host_platform == "windows":
            creation_flags = 0
            creation_flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creation_flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
            popen_kwargs["creationflags"] = creation_flags
        else:
            popen_kwargs["start_new_session"] = True
        return subprocess.Popen(command, **popen_kwargs)

    run_kwargs: dict[str, object] = {
        "check": False,
        "capture_output": True,
        "text": True,
        "env": env,
    }
    if timeout_seconds is not None:
        run_kwargs["timeout"] = max(timeout_seconds + 60, 120)
    return subprocess.run(command, **run_kwargs)
