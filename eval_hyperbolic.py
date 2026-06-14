"""
cube:
MUJOCO_GL=egl python eval_hyperbolic.py --config-name=hyperbolic_cube policy=./ogbench/Ablation/No_SigReg/lewm_hyperbolic_no_sigreg_epoch_100 solver.device=auto planning.hyperbolic.terminal_weight=0.0 planning.hyperbolic.best_weight=1.0 planning.hyperbolic.mean_weight=0.0 planning.hyperbolic.progress_weight=0.0 planning.hyperbolic.cone_weight=0.0

MUJOCO_GL=egl python eval_hyperbolic.py \
  --config-name=hyperbolic_pusht \
  policy=/data_nvme/user/zliu681/le-wm-main/lewm_cache/pusht/hyperbolic_pusht_stable_trial2/lewm_hyperbolic_pusht_stable_epoch_10 \
  solver.device=auto \
  eval.save_video=True


antmaze:
MUJOCO_GL=egl python eval_hyperbolic.py --config-name=hyperbolic_antmaze policy=./ogbench/Experiment/hyperbolic_exp_antmaze/lewm_hyperbolic_epoch_100 solver.device=auto planning.hyperbolic.terminal_weight=0.0 planning.hyperbolic.best_weight=1.0 planning.hyperbolic.mean_weight=0.0 planning.hyperbolic.progress_weight=0.0 planning.hyperbolic.cone_weight=0.0

"""

import os


def _configure_mujoco_backend() -> str:
    backend = os.environ.get("MUJOCO_GL", "egl").strip().lower() or "egl"
    os.environ["MUJOCO_GL"] = backend
    if backend in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    return backend


MUJOCO_GL_BACKEND = _configure_mujoco_backend()

from mujoco_cleanup import suppress_mujoco_egl_cleanup_errors

suppress_mujoco_egl_cleanup_errors()

import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm
import stable_worldmodel as swm

from module import ARPredictor, Embedder, MLP, SIGReg
from train_hyperbolic import (
    AdaptiveEntailmentConeLoss,
    HyperbolicJEPA,
    LorentzContrastiveLoss,
    LorentzManifold,
    build_hyperbolic_world_model,
    ensure_hyperbolic_defaults,
)
from trajectory_reachability import load_trm_checkpoint
from utils import resolve_runtime_device


def _maybe_register_ogbench_envs() -> None:
    try:
        import ogbench  # noqa: F401
        print("[eval] imported ogbench to register Gym environments", flush=True)
    except Exception as exc:
        print(
            f"[eval] ogbench import skipped: {type(exc).__name__}: {exc}",
            flush=True,
        )


def _build_env_candidates(env_name: str) -> list[str]:
    raw_candidates = [env_name]

    if env_name.startswith("visual-"):
        raw_candidates.append(env_name.removeprefix("visual-"))

    for suffix in ("-navigate-v0", "-oraclerep-v0"):
        expanded = []
        for candidate in raw_candidates:
            if candidate.endswith(suffix):
                expanded.append(candidate[: -len(suffix)] + "-v0")
        raw_candidates.extend(expanded)

    expanded = []
    for candidate in raw_candidates:
        if candidate.startswith("visual-"):
            expanded.append(candidate.removeprefix("visual-"))
    raw_candidates.extend(expanded)

    candidates = []
    seen = set()
    for candidate in raw_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def _resolve_world_env_name(env_name: str) -> str:
    _maybe_register_ogbench_envs()

    registry_keys = set(gym.registry.keys())
    if env_name in registry_keys:
        return env_name

    candidates = _build_env_candidates(env_name)
    for candidate in candidates:
        if candidate in registry_keys:
            print(
                f"[eval] resolved env_name '{env_name}' -> '{candidate}'",
                flush=True,
            )
            return candidate

    print(
        f"[eval] could not resolve env_name '{env_name}'. "
        f"Tried: {candidates}",
        flush=True,
    )
    return env_name

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def _get_column_row_count(dataset, col_name: str) -> int | None:
    h5_file = getattr(dataset, "h5_file", None)
    try:
        if h5_file is not None and col_name in h5_file:
            shape = getattr(h5_file[col_name], "shape", None)
            if shape:
                return int(shape[0])
    except Exception:
        pass

    try:
        return int(len(dataset.get_col_data(col_name)))
    except Exception:
        return None


def _resolve_safe_row_limit(dataset, extra_columns: Sequence[str]) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {"__len__": int(len(dataset))}
    episode_meta_columns = {"ep_len", "ep_offset", "episode_ends"}
    columns = [
        col for col in list(getattr(dataset, "_keys", [])) + list(extra_columns)
        if col not in episode_meta_columns
    ]

    seen = set()
    for col_name in columns:
        if col_name in seen:
            continue
        seen.add(col_name)
        count = _get_column_row_count(dataset, col_name)
        if count is not None:
            counts[col_name] = count

    safe_limit = min(counts.values())
    return safe_limit, counts


def _chunk_has_content(episode_chunk: dict[str, Any]) -> tuple[bool, str | None]:
    for col_name, value in episode_chunk.items():
        if not isinstance(value, (torch.Tensor, np.ndarray)):
            continue
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) < 1:
            continue
        if int(shape[0]) == 0:
            return False, col_name
    return True, None


def _select_valid_eval_rows(
    dataset,
    candidate_indices: np.ndarray,
    episode_idx_col: np.ndarray,
    step_idx_col: np.ndarray,
    goal_offset_steps: int,
    num_eval: int,
) -> np.ndarray:
    selected: list[int] = []
    skipped_empty = 0

    for row_idx in candidate_indices.tolist():
        ep_id = int(episode_idx_col[row_idx])
        start_step = int(step_idx_col[row_idx])
        end_step = int(start_step + goal_offset_steps)

        try:
            chunk = dataset.load_chunk(
                np.array([ep_id]),
                np.array([start_step]),
                np.array([end_step]),
            )
        except Exception as exc:
            print(
                f"[eval] skipping candidate row={row_idx} episode={ep_id} start={start_step} "
                f"because load_chunk failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        if not chunk:
            skipped_empty += 1
            continue

        has_content, bad_col = _chunk_has_content(chunk[0])
        if not has_content:
            if skipped_empty < 5:
                print(
                    f"[eval] skipping empty candidate row={row_idx} episode={ep_id} "
                    f"start={start_step} bad_col={bad_col}",
                    flush=True,
                )
            skipped_empty += 1
            continue

        selected.append(int(row_idx))
        if len(selected) >= num_eval:
            break

    if skipped_empty:
        print(f"[eval] skipped {skipped_empty} empty/invalid evaluation candidates.", flush=True)

    if len(selected) < num_eval:
        raise ValueError(
            f"Only found {len(selected)} valid evaluation starts after load_chunk validation, "
            f"but {num_eval} were requested."
        )

    return np.asarray(selected, dtype=np.int64)


def _register_hyperbolic_main_aliases():
    import __main__ as main_mod

    for obj in (
        HyperbolicJEPA,
        LorentzManifold,
        LorentzContrastiveLoss,
        AdaptiveEntailmentConeLoss,
        ARPredictor,
        Embedder,
        MLP,
        SIGReg,
    ):
        setattr(main_mod, obj.__name__, obj)

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(
            [
                HyperbolicJEPA,
                LorentzManifold,
                LorentzContrastiveLoss,
                AdaptiveEntailmentConeLoss,
                ARPredictor,
                Embedder,
                MLP,
                SIGReg,
            ]
        )


def _resolve_policy_artifact_candidates(policy_name: str, cache_dir: Path) -> list[tuple[Path, Path]]:
    raw = Path(policy_name)
    candidates = []

    def add_exact_prefix_artifacts(path_like: Path):
        path_text = str(path_like)
        for suffix in ("_state.ckpt", "_object.ckpt", "_weights.ckpt"):
            if path_text.endswith(suffix):
                add_exact_prefix_artifacts(Path(path_text[: -len(suffix)]))
                return
        if path_text.endswith(".ckpt"):
            candidates.append(path_like)
            return

        # Prefer device-neutral epoch state dicts when they exist. Object
        # checkpoints saved on Ascend may require torch_npu to unpickle.
        candidates.append(Path(f"{path_like}_state.ckpt"))
        candidates.append(Path(f"{path_like}_object.ckpt"))
        candidates.append(Path(f"{path_like}_weights.ckpt"))

    def add_parent_weights(path_like: Path):
        parent = path_like.parent
        if parent.is_dir():
            weights = sorted(parent.glob("*_weights.ckpt"))
            if len(weights) == 1:
                candidates.append(weights[0])

    add_exact_prefix_artifacts(raw)
    add_exact_prefix_artifacts(cache_dir / raw)

    if raw.is_file():
        candidates.append(raw)
        add_parent_weights(raw)
    cache_raw = cache_dir / raw
    if cache_raw.is_file():
        candidates.append(cache_raw)
        add_parent_weights(cache_raw)
    if cache_raw.is_dir():
        weights = sorted(cache_raw.glob("*_weights.ckpt"))
        if len(weights) == 1:
            candidates.append(weights[0])

    add_parent_weights(raw)
    add_parent_weights(cache_raw)

    seen = set()
    resolved = []
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            config_path = candidate.parent / "config.yaml"
            if config_path.is_file():
                resolved.append((candidate, config_path))

    if resolved:
        return resolved

    raise FileNotFoundError(
        f"Could not resolve checkpoint/config for policy '{policy_name}' under cache '{cache_dir}'."
    )


def _resolve_policy_artifacts(policy_name: str, cache_dir: Path) -> tuple[Path, Path]:
    return _resolve_policy_artifact_candidates(policy_name, cache_dir)[0]


def _should_prefer_hyperbolic_loader(cfg: DictConfig) -> bool:
    checkpoint_cfg = cfg.get("checkpoint", {})
    if bool(getattr(checkpoint_cfg, "force_rebuild", False)):
        return True

    policy_name = str(cfg.get("policy", "") or "")
    if "hyperbolic" in policy_name.lower():
        return True

    return False


def _load_hyperbolic_policy_model(cfg: DictConfig, dataset, runtime_device: str):
    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    checkpoint_candidates = _resolve_policy_artifact_candidates(cfg.policy, cache_dir)
    _register_hyperbolic_main_aliases()
    failures = []

    for checkpoint_path, config_path in checkpoint_candidates:
        print(
            f"[eval] trying hyperbolic checkpoint={checkpoint_path} config={config_path}",
            flush=True,
        )
        train_cfg = OmegaConf.load(config_path)
        ensure_hyperbolic_defaults(train_cfg)
        action_dim = int(getattr(train_cfg.wm, "action_dim", dataset.get_dim("action")))
        model = build_hyperbolic_world_model(train_cfg, action_dim=action_dim)

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            else:
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif isinstance(checkpoint, dict):
                    state_dict = checkpoint
                else:
                    raise TypeError(
                        f"Unsupported checkpoint type '{type(checkpoint).__name__}' for '{checkpoint_path}'."
                    )

                if any(key.startswith("model.") for key in state_dict.keys()):
                    state_dict = {
                        key[len("model."):]: value
                        for key, value in state_dict.items()
                        if key.startswith("model.")
                    }

                strict = bool(getattr(cfg.get("checkpoint", {}), "strict", True))
                missing, unexpected = model.load_state_dict(state_dict, strict=strict)
                print(
                    f"[eval] loaded hyperbolic checkpoint {checkpoint_path} "
                    f"(strict={strict}, missing={len(missing)}, unexpected={len(unexpected)})",
                    flush=True,
                )
                if missing:
                    print(f"[eval] missing keys: {missing}", flush=True)
                if unexpected:
                    print(f"[eval] unexpected keys: {unexpected}", flush=True)
        except ModuleNotFoundError as exc:
            if exc.name == "torch_npu" or str(exc.name).startswith("torch_npu."):
                failures.append(
                    f"{checkpoint_path}: requires torch_npu during object deserialization"
                )
                print(
                    f"[eval] checkpoint {checkpoint_path} requires torch_npu; "
                    "trying the next portable/state checkpoint candidate.",
                    flush=True,
                )
                continue
            raise
        except RuntimeError as exc:
            failures.append(f"{checkpoint_path}: {type(exc).__name__}: {exc}")
            print(
                f"[eval] checkpoint {checkpoint_path} did not match the rebuilt model; "
                "trying the next checkpoint candidate.",
                flush=True,
            )
            continue

        model = model.to(runtime_device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        return model

    failure_details = "\n".join(f"  - {failure}" for failure in failures)
    raise RuntimeError(
        "Unable to load a hyperbolic policy checkpoint on this runtime. "
        "For an object checkpoint saved from NPU training, evaluate in a torch_npu "
        "environment once or train/export a matching '<prefix>_state.ckpt' portable state dict.\n"
        f"Tried:\n{failure_details}"
    )


def load_policy_model(cfg: DictConfig, dataset, runtime_device: str):
    if _should_prefer_hyperbolic_loader(cfg):
        print(
            f"[eval] bypassing AutoCostModel and preferring rebuilt hyperbolic weights for "
            f"policy='{cfg.policy}'",
            flush=True,
        )
        return _load_hyperbolic_policy_model(cfg, dataset, runtime_device)

    try:
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to(runtime_device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        print(f"[eval] loaded policy via AutoCostModel: {cfg.policy}", flush=True)
        return model
    except Exception as exc:
        print(
            f"[eval] AutoCostModel failed for '{cfg.policy}': {type(exc).__name__}: {exc}",
            flush=True,
        )
        print("[eval] falling back to hyperbolic checkpoint loader", flush=True)
        return _load_hyperbolic_policy_model(cfg, dataset, runtime_device)


def _resolve_trm_checkpoint_path(checkpoint_name: str, cache_dir: Path) -> Path:
    raw_path = Path(checkpoint_name).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(Path(hydra.utils.to_absolute_path(str(raw_path))))
        candidates.append(cache_dir / raw_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve TRM checkpoint. Tried: {tried}")


def _maybe_attach_trm_metric(model, cfg: DictConfig, runtime_device: str) -> None:
    planning_node = cfg.get("planning")
    trm_node = planning_node.get("trm") if planning_node is not None else None
    if trm_node is None or not bool(trm_node.get("enabled", False)):
        return
    if not hasattr(model, "set_trajectory_reachability_metric"):
        raise TypeError(
            "planning.trm.enabled=True requires a model with "
            "set_trajectory_reachability_metric()."
        )

    checkpoint_name = trm_node.get("checkpoint")
    if not checkpoint_name:
        raise ValueError("planning.trm.enabled=True requires planning.trm.checkpoint.")
    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    checkpoint_path = _resolve_trm_checkpoint_path(str(checkpoint_name), cache_dir)
    metric, metadata = load_trm_checkpoint(checkpoint_path)
    metric = metric.to(runtime_device).eval()
    metric.requires_grad_(False)
    model.set_trajectory_reachability_metric(metric)
    print(
        f"[eval] loaded TRM checkpoint={checkpoint_path} "
        f"input_space={metric.input_space} max_horizon={metadata.get('max_horizon', '<unknown>')} "
        f"mode={trm_node.get('mode', 'hybrid')} weight={trm_node.get('weight', 0.5)}",
        flush=True,
    )


def _wrap_step_with_static_info(env, static_info: dict[str, Any]) -> None:
    """Inject dataset goal info into env.step infos for envs that do not emit it."""
    env._lewm_static_step_info = static_info
    if getattr(env, "_lewm_static_step_info_wrapped", False):
        return

    original_step = env.step

    def step_with_static_info(action):
        result = original_step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            info = {} if info is None else dict(info)
            for key, value in getattr(env, "_lewm_static_step_info", {}).items():
                info.setdefault(key, deepcopy(value))
            return obs, reward, terminated, truncated, info

        obs, reward, done, info = result
        info = {} if info is None else dict(info)
        for key, value in getattr(env, "_lewm_static_step_info", {}).items():
            info.setdefault(key, deepcopy(value))
        return obs, reward, done, info

    env.step = step_with_static_info
    env._lewm_static_step_info_wrapped = True


def _iter_vector_env_lists(root) -> list:
    env_lists = []
    queue = [root]
    seen = set()
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))

        envs = getattr(obj, "envs", None)
        if isinstance(envs, (list, tuple)) and envs:
            env_lists.append(envs)

        for attr in ("env", "unwrapped"):
            try:
                child = getattr(obj, attr, None)
            except Exception:
                child = None
            if child is not None and id(child) not in seen:
                queue.append(child)
    return env_lists


def _prepare_static_info_for_env(goal_step: dict[str, np.ndarray], env_index: int) -> dict[str, Any]:
    static_info = {}
    for key, value in goal_step.items():
        if not isinstance(value, np.ndarray) or value.shape[0] <= env_index:
            continue
        env_value = value[env_index]
        if isinstance(env_value, np.ndarray) and env_value.ndim > 0:
            env_value = env_value[-1]
        static_info[key] = deepcopy(env_value)
    return static_info


def _inject_static_goal_info_into_envs(world, goal_step: dict[str, np.ndarray]) -> None:
    """Make dataset goals visible in per-step env info for wrappers that require them."""
    wrapped_count = 0
    for envs in _iter_vector_env_lists(world.envs):
        if len(envs) != world.num_envs:
            continue
        for env_index, env in enumerate(envs):
            static_info = _prepare_static_info_for_env(goal_step, env_index)
            if not static_info:
                continue

            current_env = env
            seen = set()
            while current_env is not None and id(current_env) not in seen:
                seen.add(id(current_env))
                _wrap_step_with_static_info(current_env, static_info)
                wrapped_count += 1
                current_env = getattr(current_env, "env", None)

    if wrapped_count == 0:
        print("[eval] warning: did not find env wrappers to inject static goal info", flush=True)
    else:
        print(f"[eval] injected static goal info into {wrapped_count} env wrapper(s)", flush=True)


def _set_dataset_goal_xy(world, goal_step: dict[str, np.ndarray]) -> None:
    """Align OGBench MazeEnv success checks with the dataset goal state."""
    goal_qpos = goal_step.get("goal_qpos")
    if goal_qpos is None:
        return

    applied = 0
    for i, env in enumerate(_get_unwrapped_envs(world)):
        env_unwrapped = env.unwrapped
        set_goal = getattr(env_unwrapped, "set_goal", None)
        if not callable(set_goal):
            continue
        goal_xy = np.asarray(goal_qpos[i], dtype=np.float64).reshape(-1)[:2]
        if goal_xy.shape[0] != 2:
            continue
        try:
            set_goal(goal_xy=goal_xy)
            applied += 1
        except TypeError:
            continue

    if applied:
        print(f"[eval] set dataset goal_xy for {applied} env(s)", flush=True)


def _get_unwrapped_envs(world):
    envs = getattr(getattr(world.envs, "unwrapped", world.envs), "envs", None)
    if isinstance(envs, (list, tuple)) and envs:
        return envs
    for envs in _iter_vector_env_lists(world.envs):
        if len(envs) == world.num_envs:
            return envs
    return []


def _set_vector_autoreset_flags(world) -> None:
    for obj in (world.envs, getattr(world.envs, "unwrapped", None)):
        if obj is not None and hasattr(obj, "_autoreset_envs"):
            obj._autoreset_envs = np.zeros((world.num_envs,))
            return


def _resize_hwc_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    if frames.shape[-3:-1] == (height, width):
        return frames.astype(np.uint8, copy=False)

    import torch.nn.functional as F

    tensor = torch.as_tensor(frames)
    original_shape = tensor.shape
    tensor = tensor.reshape(-1, *original_shape[-3:]).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    tensor = tensor.clamp(0, 255).byte().permute(0, 2, 3, 1)
    return tensor.reshape(*original_shape[:-3], height, width, original_shape[-1]).cpu().numpy()


def _compose_rollout_grid_frame(
    rollout_frame: np.ndarray,
    target_frame: np.ndarray,
    goal_frame: np.ndarray,
) -> np.ndarray:
    """Match the four-panel layout used by rollout videos."""
    rollout_and_target = np.vstack([rollout_frame, target_frame])
    repeated_goal = np.vstack([goal_frame, goal_frame])
    return np.hstack([rollout_and_target, repeated_goal])


def evaluate_from_dataset_with_progress(
    world,
    dataset: Any,
    episodes_idx: Sequence[int],
    start_steps: Sequence[int],
    goal_offset_steps: int,
    eval_budget: int,
    callables: list[dict] | None = None,
    save_video: bool = True,
    save_outcome_frame: bool = True,
    save_grid_terminated_frame: bool = False,
    video_path: str | Path = "./",
):
    """Copy of stable_worldmodel's dataset evaluation with tqdm progress bars."""
    assert (
        world.envs.envs[0].spec.max_episode_steps is None
        or world.envs.envs[0].spec.max_episode_steps >= goal_offset_steps
    ), "env max_episode_steps must be greater than eval_budget"

    ep_idx_arr = np.array(episodes_idx)
    start_steps_arr = np.array(start_steps)
    end_steps = start_steps_arr + goal_offset_steps

    if len(ep_idx_arr) != len(start_steps_arr):
        raise ValueError("episodes_idx and start_steps must have the same length")
    if len(ep_idx_arr) != world.num_envs:
        raise ValueError("Number of episodes to evaluate must match number of envs")

    data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)
    columns = dataset.column_names

    init_step_per_env: dict[str, list[Any]] = defaultdict(list)
    goal_step_per_env: dict[str, list[Any]] = defaultdict(list)
    for ep in data:
        for col in columns:
            if col.startswith("goal"):
                continue
            if col.startswith("pixels"):
                ep[col] = ep[col].permute(0, 2, 3, 1)
            if not isinstance(ep[col], (torch.Tensor, np.ndarray)):
                continue

            init_data = ep[col][0]
            goal_data = ep[col][-1]
            if not isinstance(init_data, (np.ndarray, torch.Tensor)):
                continue

            init_data = init_data.numpy() if isinstance(init_data, torch.Tensor) else init_data
            goal_data = goal_data.numpy() if isinstance(goal_data, torch.Tensor) else goal_data
            init_step_per_env[col].append(init_data)
            goal_step_per_env[col].append(goal_data)

    init_step = {k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()}
    goal_step = {}
    for key, value in goal_step_per_env.items():
        goal_key = "goal" if key == "pixels" else f"goal_{key}"
        goal_step[goal_key] = np.stack(value)
    if "goal" in goal_step and "goal_rendered" not in goal_step:
        goal_step["goal_rendered"] = goal_step["goal"]

    seeds = init_step.get("seed")
    vkey = "variation."
    variations_dict = {
        k.removeprefix(vkey): v for k, v in init_step.items() if k.startswith(vkey)
    }
    options = [{} for _ in range(world.num_envs)]
    if variations_dict:
        for i in range(world.num_envs):
            options[i]["variation"] = list(variations_dict.keys())
            options[i]["variation_values"] = {k: v[i] for k, v in variations_dict.items()}

    init_step.update(deepcopy(goal_step))
    world.reset(seed=seeds, options=options)

    callables = callables or []
    for i, env in enumerate(_get_unwrapped_envs(world)):
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                continue
            method = getattr(env_unwrapped, method_name)
            args = spec.get("args", spec)

            prepared_args = {}
            for args_name, args_data in args.items():
                value = args_data.get("value", None)
                is_in_dataset = args_data.get("in_dataset", True)
                if is_in_dataset:
                    if value not in init_step:
                        continue
                    prepared_args[args_name] = deepcopy(init_step[value][i])
                else:
                    prepared_args[args_name] = value

            method(**prepared_args)

    for i, env in enumerate(_get_unwrapped_envs(world)):
        env_unwrapped = env.unwrapped
        if "goal_state" in init_step and "goal_state" in goal_step:
            assert np.array_equal(
                init_step["goal_state"][i], goal_step["goal_state"][i]
            ), f"Goal state info does not match at reset for env {env_unwrapped}"

    _set_dataset_goal_xy(world, goal_step)

    results: dict = {
        "success_rate": 0.0,
        "episode_successes": np.zeros(len(episodes_idx)),
        "seeds": seeds,
    }

    shape_prefix = world.infos["pixels"].shape[:2]
    init_step = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]) for k, v in init_step.items()
    }
    goal_step = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:]) for k, v in goal_step.items()
    }
    _inject_static_goal_info_into_envs(world, goal_step)

    world.infos.update(deepcopy(init_step))
    world.infos.update(deepcopy(goal_step))

    if "goal" in goal_step and "goal" in world.infos:
        assert np.allclose(world.infos["goal"], goal_step["goal"]), "Goal info does not match"

    target_frames = torch.stack([ep["pixels"] for ep in data]).numpy()
    video_frames = None
    video_height = video_width = None
    first_terminated_frames: list[np.ndarray | None] = [None] * world.num_envs
    outcome_frames: list[np.ndarray | None] = [None] * world.num_envs
    first_terminated_steps = np.full(world.num_envs, -1, dtype=np.int64)

    env_label = getattr(world.envs.envs[0].spec, "id", "WorldModel Eval")
    with tqdm(total=eval_budget, desc=f"Evaluating {env_label}", unit="step") as pbar:
        for step_idx in range(eval_budget):
            current_frame = world.infos["pixels"][:, -1]
            if video_frames is None:
                video_height, video_width = map(int, current_frame.shape[-3:-1])
                video_frames = np.empty(
                    (world.num_envs, eval_budget, *current_frame.shape[-3:]),
                    dtype=np.uint8,
                )
                target_frames = _resize_hwc_frames(target_frames, video_height, video_width)
            video_frames[:, step_idx] = _resize_hwc_frames(current_frame, video_height, video_width)
            world.infos.update(deepcopy(goal_step))
            world.step()
            step_terminateds = np.asarray(world.terminateds, dtype=bool)
            newly_terminated = np.logical_and(
                np.logical_not(results["episode_successes"]),
                step_terminateds,
            )
            if np.any(newly_terminated):
                post_step_frames = _resize_hwc_frames(
                    world.infos["pixels"][:, -1],
                    video_height,
                    video_width,
                )
                target_len = target_frames.shape[1]
                for env_idx in np.flatnonzero(newly_terminated):
                    outcome_frames[env_idx] = post_step_frames[env_idx].copy()
                    first_terminated_frames[env_idx] = _compose_rollout_grid_frame(
                        post_step_frames[env_idx],
                        target_frames[env_idx, (step_idx + 1) % target_len],
                        target_frames[env_idx, -1],
                    )
                    first_terminated_steps[env_idx] = step_idx + 1
            results["episode_successes"] = np.logical_or(
                results["episode_successes"], step_terminateds
            )
            _set_vector_autoreset_flags(world)
            success_count = int(np.sum(results["episode_successes"]))
            pbar.set_postfix(success=f"{success_count}/{len(episodes_idx)}")
            pbar.update(1)

    if video_frames is None:
        current_frame = world.infos["pixels"][:, -1]
        video_height, video_width = map(int, current_frame.shape[-3:-1])
        video_frames = np.empty(
            (world.num_envs, eval_budget, *current_frame.shape[-3:]),
            dtype=np.uint8,
        )
        target_frames = _resize_hwc_frames(target_frames, video_height, video_width)
    video_frames[:, -1] = _resize_hwc_frames(world.infos["pixels"][:, -1], video_height, video_width)
    final_frames = video_frames[:, -1]
    n_episodes = len(episodes_idx)
    results["success_rate"] = float(np.sum(results["episode_successes"])) / n_episodes * 100.0
    results["first_terminated_steps"] = first_terminated_steps

    if save_video or save_outcome_frame:
        import imageio

        target_len = target_frames.shape[1]
        video_path_obj = Path(video_path)
        video_path_obj.mkdir(parents=True, exist_ok=True)
        if save_video:
            for i in tqdm(range(world.num_envs), desc="Writing videos", unit="video"):
                out = imageio.get_writer(
                    video_path_obj / f"rollout_{i}.mp4",
                    fps=15,
                    codec="libx264",
                )
                for t in range(eval_budget):
                    frame = _compose_rollout_grid_frame(
                        video_frames[i, t],
                        target_frames[i, t % target_len],
                        target_frames[i, -1],
                    )
                    out.append_data(frame)
                out.close()
                if save_grid_terminated_frame and first_terminated_frames[i] is not None:
                    imageio.imwrite(
                        video_path_obj / f"rollout_{i}_terminated_step_{first_terminated_steps[i]:03d}.png",
                        first_terminated_frames[i],
                    )
            print(f"Video saved to {video_path_obj}")
            if save_grid_terminated_frame:
                print(
                    f"Saved {sum(frame is not None for frame in first_terminated_frames)} "
                    f"first-terminated grid frame(s) to {video_path_obj}",
                    flush=True,
                )

        if save_outcome_frame:
            for i in tqdm(range(world.num_envs), desc="Writing outcome frames", unit="frame"):
                if bool(results["episode_successes"][i]):
                    frame = outcome_frames[i]
                    if frame is None:
                        frame = final_frames[i]
                    filename = f"rollout_{i}_success_step_{first_terminated_steps[i]:03d}_frame.png"
                else:
                    frame = final_frames[i]
                    filename = f"rollout_{i}_failure_last_frame.png"
                imageio.imwrite(
                    video_path_obj / filename,
                    frame.astype(np.uint8, copy=False),
                )
            print(f"Outcome frames saved to {video_path_obj}", flush=True)

    if results["seeds"] is not None:
        assert np.unique(results["seeds"]).shape[0] == n_episodes, "Some episode seeds are identical!"

    return results

@hydra.main(version_base=None, config_path="config/eval", config_name="hyperbolic")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"
    print(f"[eval] mujoco backend: {MUJOCO_GL_BACKEND}", flush=True)

    runtime_device = resolve_runtime_device(
        cfg.solver.get("device", "auto"),
        allow_fallback=True,
    )
    with open_dict(cfg):
        cfg.solver.device = runtime_device
        cfg.world.env_name = _resolve_world_env_name(str(cfg.world.env_name))

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world_image_shape = (
        int(cfg.world.get("height", cfg.eval.img_size)),
        int(cfg.world.get("width", cfg.eval.img_size)),
    )
    world = swm.World(**cfg.world, image_shape=world_image_shape)

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    process_columns = cfg.dataset.get("keys_to_process", cfg.dataset.keys_to_cache)
    for col in process_columns:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = load_policy_model(cfg, dataset, runtime_device)
        planning_node = cfg.get("planning")
        planning_cfg = (
            OmegaConf.to_container(planning_node, resolve=True)
            if planning_node is not None
            else {}
        )
        if hasattr(model, "set_planning_config"):
            model.set_planning_config(planning_cfg)
            if isinstance(model, HyperbolicJEPA):
                resolved_cfg = (
                    model._planning_hyperbolic_config()
                    if hasattr(model, "_planning_hyperbolic_config")
                    else planning_cfg.get("hyperbolic", "<defaults>")
                )
                print(
                    f"[eval] hyperbolic planning config: {resolved_cfg}",
                    flush=True,
                )
                print(
                    f"[eval] tangent stabilization config: {model.tangent_stabilization}",
                    flush=True,
                )
        _maybe_attach_trm_metric(model, cfg, runtime_device)
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    safe_row_limit, row_counts = _resolve_safe_row_limit(dataset, [col_name, "step_idx"])
    dataset_len = int(len(dataset))
    unique_counts = sorted(set(row_counts.values()))
    if len(unique_counts) > 1:
        print(
            f"[eval] warning: inconsistent row counts detected across dataset columns: "
            f"{row_counts}. Dataset length is {dataset_len}; row-wise safe limit is {safe_row_limit}.",
            flush=True,
        )
    episode_idx_col = np.asarray(dataset.get_col_data(col_name))
    step_idx_col = np.asarray(dataset.get_col_data("step_idx"))
    if len(episode_idx_col) != len(step_idx_col):
        raise ValueError(
            f"Dataset metadata is inconsistent: len({col_name})={len(episode_idx_col)} "
            f"but len(step_idx)={len(step_idx_col)}."
        )
    episode_idx_col = episode_idx_col[:safe_row_limit]
    step_idx_col = step_idx_col[:safe_row_limit]
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in episode_idx_col])

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = step_idx_col <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    valid_indices = valid_indices[valid_indices < safe_row_limit]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f"Not enough valid starting points for evaluation: requested {cfg.eval.num_eval}, "
            f"found {len(valid_indices)} after filtering against dataset length {dataset_len}."
        )

    g = np.random.default_rng(cfg.seed)
    candidate_indices = g.permutation(valid_indices)
    random_episode_indices = np.sort(
        _select_valid_eval_rows(
            dataset,
            candidate_indices=candidate_indices,
            episode_idx_col=episode_idx_col,
            step_idx_col=step_idx_col,
            goal_offset_steps=cfg.eval.goal_offset_steps,
            num_eval=cfg.eval.num_eval,
        )
    )

    print(random_episode_indices)

    eval_episodes = episode_idx_col[random_episode_indices]
    eval_start_idx = step_idx_col[random_episode_indices]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = evaluate_from_dataset_with_progress(
        world,
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        save_video=bool(cfg.eval.get("save_video", True)),
        save_outcome_frame=bool(cfg.eval.get("save_outcome_frame", True)),
        save_grid_terminated_frame=bool(cfg.eval.get("save_grid_terminated_frame", False)),
        video_path=results_path,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
