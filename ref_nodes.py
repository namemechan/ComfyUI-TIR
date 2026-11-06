"""Text-inline SnapRef routing nodes for ComfyUI."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.patcher_extension
import folder_paths


logger = logging.getLogger("TIR.Reference")

_TAG_PATTERN = re.compile(r"<ref:([^<>:]+)(?::([^<>]*))?>", re.IGNORECASE)
_LOOKALIKE_PATTERN = re.compile(r"<ref:[^<>]*>", re.IGNORECASE)
_IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".gif", ".tif", ".tiff",
}
_KNOWN_KEYS = {"g", "e", "c", "mp", "w", "m"}
_WIRELESS_STORE: dict[str, dict] = {}
_WIRELESS_LOCK = threading.Lock()


def _log(enabled: bool, message: str, *, warning: bool = False) -> None:
    if not enabled:
        return
    (logger.warning if warning else logger.info)(message)


def _fingerprint_prompt(prompt) -> str | None:
    if not prompt:
        return None
    try:
        data = json.dumps(prompt, sort_keys=True, default=str)
    except Exception:
        return None
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def parse_references(text: str, log: bool = True) -> tuple[list[dict], str]:
    """Return parsed calls and text with recognized call syntax removed."""
    refs: list[dict] = []
    spans: list[tuple[int, int]] = []
    matched = set()

    for match in _TAG_PATTERN.finditer(text or ""):
        name = match.group(1).strip()
        options: dict[str, str] = {}
        raw_options = match.group(2) or ""
        option_items = raw_options.split(":") if raw_options else ()
        for item in option_items:
            item = item.strip()
            if not item:
                continue
            if item.count("=") != 1:
                _log(log, f"[TIR Ref] 잘못된 옵션 '{item}'을 무시합니다. (예: g=4.5)", warning=True)
                continue
            key, value = (part.strip() for part in item.split("=", 1))
            key = key.lower()
            if key not in _KNOWN_KEYS:
                _log(log, f"[TIR Ref] 알 수 없는 옵션 '{key}'을 무시합니다.", warning=True)
                continue
            if not value:
                _log(log, f"[TIR Ref] 옵션 '{key}' 값이 비어 있어 무시합니다.", warning=True)
                continue
            if key in options:
                _log(log, f"[TIR Ref] 옵션 '{key}'이 중복되어 마지막 값을 사용합니다.", warning=True)
            options[key] = value

        refs.append({"image": name, "options": options})
        spans.append(match.span())
        matched.add(match.group(0))

    if log:
        for match in _LOOKALIKE_PATTERN.finditer(text or ""):
            if match.group(0) not in matched:
                _log(log, f"[TIR Ref] 호출 형식으로 보이지만 인식되지 않아 남겨둡니다: {match.group(0)}", warning=True)

    if not spans:
        return refs, text
    pieces = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return refs, "".join(pieces)


def _float_option(options: dict, key: str, default: float, low: float, high: float, log: bool) -> float:
    if key not in options:
        return default
    try:
        value = float(options[key])
        if not math.isfinite(value):
            raise ValueError("not finite")
    except (TypeError, ValueError):
        _log(log, f"[TIR Ref] '{key}={options[key]}'은 유효한 숫자가 아니어서 기본값 {default}을 사용합니다.", warning=True)
        return default
    if value < low or value > high:
        clamped = min(high, max(low, value))
        _log(log, f"[TIR Ref] '{key}={value}'을 허용 범위 {low}–{high}로 제한합니다.", warning=True)
        return clamped
    return value


def _correction_option(options: dict, default: str, log: bool) -> str:
    """c=0 -> apg, c=1 -> rescale, c=2 -> none."""
    if "c" not in options:
        return default
    values = {"0": "apg", "1": "rescale", "2": "none"}
    value = options["c"].strip().lower()
    if value in values:
        return values[value]
    _log(log, f"[TIR Ref] 'c={value}'은 0, 1, 2 중 하나여야 하므로 기본값 '{default}'을 사용합니다.", warning=True)
    return default


def _first_settings(ref: dict, log: bool) -> dict:
    options = ref["options"]
    return {
        "g": _float_option(options, "g", 4.5, 0.0, 20.0, log),
        "g_explicit": "g" in options,
        "e": _float_option(options, "e", 1.0, 0.0, 1.0, log),
        "c": _correction_option(options, "apg", log),
        "mp": _float_option(options, "mp", 1.0, 0.25, 2.0, log),
        "m": _parse_mode(options, log),
    }


def _parse_mode(options: dict, log: bool) -> str:
    """m=0 -> Character & Style (cs), m=1 -> Character (char)."""
    if "m" not in options:
        return "cs"
    value = options["m"].strip()
    if value == "0":
        return "cs"
    if value == "1":
        return "char"
    _log(log, f"[TIR Ref] 'm={value}'은 0 또는 1이어야 하므로 기본 모드 Character & Style을 사용합니다.", warning=True)
    return "cs"


def _ordered_image_files(root: str, include_subfolders: bool) -> list[str]:
    files: list[str] = []
    if include_subfolders:
        for current, dirs, names in os.walk(root):
            dirs.sort(key=lambda s: (s.casefold(), s))
            for name in sorted(names, key=lambda s: (s.casefold(), s)):
                if Path(name).suffix.lower() in _IMAGE_EXTS:
                    files.append(os.path.join(current, name))
    else:
        try:
            names = os.listdir(root)
        except OSError:
            return []
        for name in sorted(names, key=lambda s: (s.casefold(), s)):
            path = os.path.join(root, name)
            if os.path.isfile(path) and Path(name).suffix.lower() in _IMAGE_EXTS:
                files.append(path)

    return sorted(files, key=lambda p: (os.path.relpath(p, root).replace("\\", "/").casefold(),
                                         os.path.relpath(p, root).replace("\\", "/")))


def _resolve_image(name: str, files: list[str], root: str) -> str | None:
    """Exact extension selects that file; a stem selects the first sorted match."""
    clean_name = name.strip().replace("/", os.sep).replace("\\", os.sep)
    relative = os.path.normpath(clean_name)
    if os.path.isabs(relative) or relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return None

    suffix = Path(relative).suffix.lower()
    if suffix in _IMAGE_EXTS:
        exact_relative = os.path.join(root, relative)
        allowed_paths = {os.path.normcase(os.path.abspath(path)) for path in files}
        if os.path.normcase(os.path.abspath(exact_relative)) in allowed_paths:
            return exact_relative
        base = os.path.basename(relative)
        for path in files:
            if os.path.basename(path).casefold() == base.casefold():
                return path
        return None

    requested = os.path.basename(relative).casefold()
    for path in files:
        if Path(path).stem.casefold() == requested:
            return path
    return None


def _load_image(path: str) -> torch.Tensor:
    """Load a local reference into ComfyUI IMAGE format [1, H, W, 3]."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        array = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def _node_mappings():
    import nodes
    return getattr(nodes, "NODE_CLASS_MAPPINGS", {})


def _node_class(key: str):
    cls = _node_mappings().get(key)
    if cls is None:
        raise RuntimeError(f"필요한 ComfyUI 노드 '{key}'를 찾지 못했습니다. 해당 레퍼런스 노드를 설치하고 ComfyUI를 재시작하세요.")
    return cls


def _node_choices(key: str, input_name: str, fallback_filter=None) -> list[str]:
    try:
        cls = _node_mappings().get(key)
        if cls is not None:
            choices = cls.INPUT_TYPES()["required"][input_name][0]
            if choices:
                return list(choices)
    except Exception:
        pass
    names = folder_paths.get_filename_list("loras")
    if fallback_filter:
        filtered = [name for name in names if fallback_filter(name)]
        if filtered:
            return filtered
    return list(names) or [""]


def _snapref_adapter_choices() -> list[str]:
    return _node_choices("SnapRef", "adapter", lambda name: "snapref" in name.lower() or "anima_ref_kv" in name.lower())


def _snapref_core():
    cls = _node_class("SnapRef")
    module = sys.modules.get(cls.__module__)
    if module is None:
        raise RuntimeError("SnapRef 모듈을 찾지 못했습니다. SnapRef 설치 상태를 확인하세요.")
    required = ("_Adapter", "_adapters", "_adapter_block_map", "_encode_image", "_make_wrapper", "_make_outer", "WRAPPER_KEY")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"설치된 SnapRef가 필요한 내부 기능을 제공하지 않습니다: {', '.join(missing)}")
    return module


def _apply_snapref(model, clip, vae, refs: list[torch.Tensor], weights: list[float], adapter: str, settings: dict):
    if not any(weight > 0 for weight in weights):
        return model
    if clip is None:
        raise RuntimeError("SnapRef를 사용하려면 CLIP 입력이 필요합니다.")

    core = _snapref_core()
    adapter_path = folder_paths.get_full_path_or_raise("loras", adapter)
    if adapter_path not in core._adapters:
        core._adapters[adapter_path] = core._Adapter(adapter_path)
    adapter_obj = core._adapters[adapter_path]
    core._adapter_block_map(adapter_obj.block_indices, model.get_model_object("diffusion_model"))

    kept_refs, kept_weights = [], []
    for image, weight in zip(refs, weights):
        if weight <= 0:
            continue
        latent, _size = core._encode_image(model, vae, image, settings["mp"])
        kept_refs.append(latent)
        kept_weights.append(float(weight))
    if not kept_refs:
        return model

    ref_cond = clip.encode_from_tokens_scheduled(clip.tokenize(""))[0]
    patched = model.clone()
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        core.WRAPPER_KEY,
        core._make_wrapper(adapter_obj, kept_refs, ref_cond, settings["m"], False, 0.0, kept_weights, None),
    )
    wrappers = patched.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]
    patched.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL] = {
        core.WRAPPER_KEY: wrappers.pop(core.WRAPPER_KEY), **wrappers,
    }
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        core.WRAPPER_KEY,
        core._make_outer(adapter_obj),
    )

    if settings["g_explicit"] and settings["g"] != 1.0:
        if not hasattr(core, "_make_ref_guidance"):
            raise RuntimeError("설치된 SnapRef에서 ref_guidance 기능을 찾지 못했습니다.")
        patched.set_model_sampler_post_cfg_function(core._make_ref_guidance(settings["g"], settings["c"], settings["e"]))
    return patched


def _apply_stack(model, vae, clip, ref_stack: list[dict], reference_folder: str,
                 adapter: str, include_subfolders: bool, log: bool):
    if not ref_stack:
        return model

    first = ref_stack[0]
    settings = _first_settings(first, log)
    accepted_first_options = {"g", "e", "c", "mp", "w", "m"}
    unsupported_first = set(first["options"]) - accepted_first_options
    if unsupported_first:
        _log(log, f"[TIR Ref] 사용할 수 없는 첫 호출 옵션 {', '.join(sorted(unsupported_first))}을 무시합니다.", warning=True)
    folder_text = reference_folder.strip()
    if not folder_text:
        _log(log, "[TIR Ref] 이미지 폴더 경로가 비어 있어 모델을 변경하지 않습니다.", warning=True)
        return model
    root = os.path.abspath(os.path.expanduser(folder_text))
    if not os.path.isdir(root):
        _log(log, f"[TIR Ref] 이미지 폴더를 찾을 수 없습니다: {root}", warning=True)
        return model

    files = _ordered_image_files(root, include_subfolders)
    images: list[torch.Tensor] = []
    weights: list[float] = []

    for index, ref in enumerate(ref_stack):
        if index > 0:
            ignored = set(ref["options"]) - {"w"}
            if ignored:
                _log(log, f"[TIR Ref] 뒤따르는 호출의 옵션 {', '.join(sorted(ignored))}은 첫 호출 설정을 따르므로 무시합니다.", warning=True)

        image_path = _resolve_image(ref["image"], files, root)
        if image_path is None:
            _log(log, f"[TIR Ref] 이미지를 찾지 못했습니다: {ref['image']}", warning=True)
            continue

        try:
            image = _load_image(image_path)
        except Exception as exc:
            _log(log, f"[TIR Ref] 이미지 로드 실패: {image_path} ({exc})", warning=True)
            continue

        images.append(image)
        weights.append(_float_option(ref["options"], "w", 1.0, 0.0, 2.0, log))
        _log(log, f"[TIR Ref] 이미지 선택: {os.path.relpath(image_path, root)}")

    if not images:
        _log(log, "[TIR Ref] 처리할 레퍼런스 이미지가 없어 모델을 변경하지 않습니다.", warning=True)
        return model

    if not any(weight > 0 for weight in weights):
        _log(log, "[TIR Ref] 레퍼런스 가중치가 모두 0이어서 모델을 변경하지 않습니다.", warning=True)
        return model

    if not adapter:
        raise RuntimeError("SnapRef 어댑터를 Receiver에서 선택하세요.")
    out_model = _apply_snapref(model, clip, vae, images, weights, adapter, settings)

    applied_count = sum(weight > 0 for weight in weights)
    _log(log, f"[TIR Ref] SnapRef 레퍼런스 {applied_count}개 적용 완료.")
    return out_model


class TIR_Ref_Sender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "프롬프트에 <ref:이미지이름>을 입력합니다. 옵션은 :키=값으로 추가할 수 있습니다.",
            }),
            "log": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("TIR_REF_STACK", "STRING")
    RETURN_NAMES = ("ref_stack", "text")
    FUNCTION = "run"
    CATEGORY = "TIR/Reference"

    def run(self, text, log=True):
        refs, cleaned = parse_references(text, log=log)
        _log(log, f"[TIR Ref] 호출 {len(refs)}개 파싱 완료.")
        return (refs, cleaned)


class TIR_Ref_Receiver:
    @classmethod
    def INPUT_TYPES(cls):
        adapter_choices = _snapref_adapter_choices()
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "ref_stack": ("TIR_REF_STACK",),
                "reference_folder": ("STRING", {
                    "default": folder_paths.get_input_directory(),
                    "multiline": False,
                    "tooltip": "프롬프트에서 지정한 이미지 이름을 찾을 폴더 경로",
                }),
                "adapter": (adapter_choices, {"default": adapter_choices[0]}),
                "include_subfolders": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "켜면 지정 폴더의 하위 폴더까지 이미지 검색에 포함합니다.",
                }),
                "log": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "TIR/Reference"

    def run(self, model, vae, clip, ref_stack, reference_folder, adapter,
            include_subfolders=False, log=True):
        return (_apply_stack(model, vae, clip, ref_stack, reference_folder,
                             adapter, include_subfolders, log),)


class TIR_Ref_Sender_Wireless:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "프롬프트에 <ref:이미지이름>을 입력합니다. 옵션은 :키=값으로 추가할 수 있습니다.",
                }),
                "channel": ("STRING", {"default": "TIR-REF"}),
                "log": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "TIR/Reference"

    @classmethod
    def IS_CHANGED(cls, text, channel, log, prompt=None):
        return float("nan")

    def run(self, text, channel, log=True, prompt=None):
        refs, cleaned = parse_references(text, log=log)
        with _WIRELESS_LOCK:
            _WIRELESS_STORE[channel] = {
                "refs": refs,
                "fingerprint": _fingerprint_prompt(prompt),
            }
        _log(log, f"[TIR Ref][Wireless] channel='{channel}'에 호출 {len(refs)}개를 저장했습니다.")
        return (cleaned,)


class TIR_Ref_Receiver_Wireless:
    @classmethod
    def INPUT_TYPES(cls):
        adapter_choices = _snapref_adapter_choices()
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "channel": ("STRING", {"default": "TIR-REF"}),
                "reference_folder": ("STRING", {
                    "default": folder_paths.get_input_directory(),
                    "multiline": False,
                }),
                "adapter": (adapter_choices, {"default": adapter_choices[0]}),
                "include_subfolders": ("BOOLEAN", {"default": False}),
                "log": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "TIR/Reference"

    @classmethod
    def IS_CHANGED(cls, model, vae, clip, channel, reference_folder, adapter,
                   include_subfolders=False, log=True, prompt=None):
        return float("nan")

    def run(self, model, vae, clip, channel, reference_folder, adapter,
            include_subfolders=False, log=True, prompt=None):
        with _WIRELESS_LOCK:
            entry = _WIRELESS_STORE.get(channel)
        if entry is None:
            _log(log, f"[TIR Ref][Wireless] channel='{channel}' 데이터가 없어 모델을 그대로 통과시킵니다.", warning=True)
            return (model,)

        current_fp = _fingerprint_prompt(prompt)
        stored_fp = entry.get("fingerprint")
        if current_fp is not None and stored_fp is not None and current_fp != stored_fp:
            _log(log, f"[TIR Ref][Wireless] channel='{channel}' 데이터가 이번 워크플로 실행의 것이 아니어서 무시합니다.", warning=True)
            return (model,)
        if current_fp is None or stored_fp is None:
            _log(log, f"[TIR Ref][Wireless] channel='{channel}' 실행 지문을 확인할 수 없어 검증 없이 적용합니다.")

        return (_apply_stack(model, vae, clip, entry.get("refs", []), reference_folder,
                             adapter, include_subfolders, log),)


NODE_CLASS_MAPPINGS = {
    "TIR_Ref_Sender": TIR_Ref_Sender,
    "TIR_Ref_Receiver": TIR_Ref_Receiver,
    "TIR_Ref_Sender_Wireless": TIR_Ref_Sender_Wireless,
    "TIR_Ref_Receiver_Wireless": TIR_Ref_Receiver_Wireless,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TIR_Ref_Sender": "TIR Ref - Sender",
    "TIR_Ref_Receiver": "TIR Ref - Receiver",
    "TIR_Ref_Sender_Wireless": "TIR Ref - Sender (Wireless)",
    "TIR_Ref_Receiver_Wireless": "TIR Ref - Receiver (Wireless)",
}
