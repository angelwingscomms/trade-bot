from __future__ import annotations

from .shared import *  # noqa: F401,F403


def deploy_active_model(runtime: Mt5RuntimePaths, model_dir: Path) -> None:
    """Copy the live source tree and referenced archived model into the MT5 runtime."""

    ensure_runtime_dirs(runtime)
    log = logging.getLogger(__name__)

    # Ensure subdirectory parents exist for the new folder layout.
    runtime.deployed_live_mq5.parent.mkdir(parents=True, exist_ok=True)
    runtime.deployed_compile_log.parent.mkdir(parents=True, exist_ok=True)
    runtime.deployed_live_ex5.parent.mkdir(parents=True, exist_ok=True)

    # Sync the entire live/ source tree (live.mq5 + functions/) so that the
    # MQL5 compiler can resolve all relative #include paths.
    live_src = LIVE_MQ5_PATH.parent  # repo: live/
    live_dst = runtime.deployed_live_mq5.parent  # expert: <project>/live/
    sync_directory_contents(live_src, live_dst)
    log.debug("Synced live/ source tree to %s", live_dst)

    relative_model_dir = model_dir.relative_to(ROOT_DIR)
    runtime_model_dir = runtime.expert_dir / relative_model_dir
    runtime_model_dir.mkdir(parents=True, exist_ok=True)
    _copy_with_retries(
        model_onnx_path(model_dir), runtime_model_dir / MODEL_FILE_NAME, log=log
    )
    _copy_with_retries(
        model_config_path(model_dir), runtime_model_dir / MODEL_CONFIG_NAME, log=log
    )
