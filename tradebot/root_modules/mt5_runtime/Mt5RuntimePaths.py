from __future__ import annotations

from .shared import *  # noqa: F401,F403


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
        return self.expert_dir / "live" / "live.mq5"

    @property
    def deployed_live_ex5(self) -> Path:
        return self.expert_dir / "mt5" / "compiled" / "live.ex5"

    @property
    def deployed_compile_log(self) -> Path:
        return self.expert_dir / "mt5" / "logs" / "live.compile.log"

    @property
    def deployed_model_path(self) -> Path:
        return self.expert_dir / "artifacts" / "model.onnx"

    @property
    def expert_resource_name(self) -> str:
        return f"{PROJECT_DIR_NAME}\\mt5\\compiled\\live.ex5"
