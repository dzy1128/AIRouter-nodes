import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
import torch
from PIL import Image


ASPECT_RATIO_OPTIONS = [
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "16:10",
    "10:16",
    "21:9",
    "9:21",
]

IMAGE_SIZE_OPTIONS = [
    "1K",
    "2K",
    "3K",
    "4K",
]


def _placeholder_image(size: int = 512) -> torch.Tensor:
    return torch.zeros((1, size, size, 3), dtype=torch.float32)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.dim() == 4:
        image = image[0]
    array = image.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.shape[-1] == 4:
        return Image.fromarray(array, mode="RGBA").convert("RGB")
    return Image.fromarray(array, mode="RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _tensor_to_data_url(image: torch.Tensor) -> str:
    pil_image = _tensor_to_pil(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _tensor_to_inline_data(image: torch.Tensor) -> Dict[str, str]:
    pil_image = _tensor_to_pil(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "mimeType": "image/png",
        "data": encoded,
    }


def _decode_base64_payload(payload: str) -> bytes:
    if payload.startswith("data:") and ";base64," in payload:
        payload = payload.split(";base64,", 1)[1]
    return base64.b64decode(payload)


def _format_response_body(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(payload)


def _extract_texts(payload: Any) -> List[str]:
    texts: List[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "text" and isinstance(value, str) and value.strip():
                texts.append(value.strip())
            else:
                texts.extend(_extract_texts(value))
    elif isinstance(payload, list):
        for item in payload:
            texts.extend(_extract_texts(item))
    return texts


def _normalize_data_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _collect_image_items(payload: Any) -> List[Dict[str, Any]]:
    image_items: List[Dict[str, Any]] = []
    candidate_string_keys = (
        "base64",
        "b64_json",
        "bytes",
        "url",
        "image",
        "data",
        "image_url",
        "imageUrl",
        "uri",
        "image_uri",
        "imageUri",
    )

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if any(
                key in node and isinstance(node.get(key), str) and node.get(key)
                for key in candidate_string_keys
            ):
                image_items.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return image_items


def _get_nested_inline_data(node: Dict[str, Any]) -> Optional[str]:
    for key in ("inline_data", "inlineData"):
        value = node.get(key)
        if isinstance(value, dict):
            data_value = value.get("data") or value.get("base64") or value.get("b64_json")
            if isinstance(data_value, str) and data_value:
                return data_value
    return None


def _summarize_response_payload(payload: Any) -> List[str]:
    lines: List[str] = []

    if isinstance(payload, dict):
        top_keys = ", ".join(sorted(str(key) for key in payload.keys())[:20])
        if top_keys:
            lines.append(f"响应顶层字段: {top_keys}")

        data = payload.get("data")
        if isinstance(data, list):
            lines.append(f"data 数量: {len(data)}")
            if data and isinstance(data[0], dict):
                item_keys = ", ".join(sorted(str(key) for key in data[0].keys())[:20])
                if item_keys:
                    lines.append(f"data[0] 字段: {item_keys}")
        elif isinstance(data, dict):
            item_keys = ", ".join(sorted(str(key) for key in data.keys())[:20])
            if item_keys:
                lines.append(f"data 字段: {item_keys}")

    image_items = _collect_image_items(payload)
    lines.append(f"递归检测到图片候选数: {len(image_items)}")
    if image_items:
        first_item_keys = ", ".join(sorted(str(key) for key in image_items[0].keys())[:20])
        if first_item_keys:
            lines.append(f"首个图片候选字段: {first_item_keys}")

    return lines


def _get_api_key() -> str:
    for env_name in ("AIROUTER-API-KEY", "AIROUTER_API_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return ""


def _is_gemini_model(model: str) -> bool:
    return model.strip().lower().startswith("gemini")


def _validate_request_params(
    *,
    model: str,
    aspect_ratio: str,
    image_size: str,
    response_format: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    timeout_seconds: int,
    base_url: str,
    seed: int,
    input_image_count: int,
) -> Optional[str]:
    if not model.strip():
        return "非法参数 model：模型名不能为空。"
    if aspect_ratio not in ASPECT_RATIO_OPTIONS:
        return f"非法参数 aspect_ratio：{aspect_ratio}，可选值：{', '.join(ASPECT_RATIO_OPTIONS)}。"
    if image_size not in IMAGE_SIZE_OPTIONS:
        return f"非法参数 image_size：{image_size}，可选值：{', '.join(IMAGE_SIZE_OPTIONS)}。"
    if response_format.strip() not in ("url", "base64", "bytes", "json", ""):
        return f"非法参数 response_format：{response_format}。"
    if not (0.0 <= float(temperature) <= 2.0):
        return f"非法参数 temperature：{temperature}，范围应为 0.0 到 2.0。"
    if not (0.0 <= float(top_p) <= 1.0):
        return f"非法参数 top_p：{top_p}，范围应为 0.0 到 1.0。"
    if not (1 <= int(max_output_tokens) <= 262144):
        return f"非法参数 max_output_tokens：{max_output_tokens}，范围应为 1 到 262144。"
    if not (10 <= int(timeout_seconds) <= 600):
        return f"非法参数 timeout_seconds：{timeout_seconds}，范围应为 10 到 600。"
    if not base_url.strip():
        return "非法参数 base_url：不能为空。"
    if not (0 <= int(seed) <= 0xFFFFFFFFFFFFFFFF):
        return f"非法参数 seed：{seed}，范围应为 0 到 18446744073709551615。"
    if input_image_count < 0 or input_image_count > 5:
        return f"非法参数 输入图片数量：{input_image_count}，允许范围为 0 到 5。"
    return None


def _extract_invalid_field_hint(message: str) -> str:
    if not message:
        return ""

    patterns = [
        r"Invalid value at '([^']+)'",
        r"invalid value at '([^']+)'",
        r"Field '([^']+)'",
        r"field '([^']+)'",
        r"parameter '([^']+)'",
        r"参数[:：]\s*([A-Za-z0-9_.\[\]-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return f"非法参数 {match.group(1)}"
    return ""


class AIRouterImageBase:
    NODE_TITLE = "AIRouter Image"
    DEFAULT_MODEL = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "输入提示词。图生图时可以留空，但建议配合提示词使用。",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": cls.DEFAULT_MODEL,
                        "multiline": False,
                    },
                ),
                "aspect_ratio": (
                    ASPECT_RATIO_OPTIONS,
                    {
                        "default": "1:1",
                    },
                ),
                "image_size": (
                    IMAGE_SIZE_OPTIONS,
                    {
                        "default": "1K",
                    },
                ),
                "response_format": (
                    "STRING",
                    {
                        "default": "url",
                        "multiline": False,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "max_output_tokens": (
                    "INT",
                    {
                        "default": 32768,
                        "min": 1,
                        "max": 262144,
                        "step": 1,
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 180,
                        "min": 10,
                        "max": 600,
                        "step": 1,
                    },
                ),
                "base_url": (
                    "STRING",
                    {
                        "default": "https://api-ai.gk.cn",
                        "multiline": False,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "log")
    FUNCTION = "generate"
    CATEGORY = "AIRouter"

    def _build_payload(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        image_size: str,
        response_format: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        input_images: Sequence[torch.Tensor],
    ) -> Dict[str, Any]:
        input_payload: Dict[str, Any] = {}
        if prompt.strip():
            input_payload["prompt"] = prompt.strip()
        if input_images:
            input_payload["images"] = [_tensor_to_data_url(image) for image in input_images]

        return {
            "model": model.strip(),
            "input": input_payload,
            "parameters": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["TEXT", "IMAGE"],
                "topP": top_p,
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                },
            },
            "format": response_format.strip(),
        }

    def _build_gemini_payload(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        image_size: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        seed: int,
        input_images: Sequence[torch.Tensor],
    ) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = []
        if prompt.strip():
            parts.append({"text": prompt.strip()})
        for image in input_images:
            parts.append({"inlineData": _tensor_to_inline_data(image)})

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts or [{"text": ""}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_output_tokens,
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                },
            },
        }

        if seed:
            payload["generationConfig"]["seed"] = seed

        return payload

    def _request_images(
        self,
        payload: Dict[str, Any],
        base_url: str,
        timeout_seconds: int,
    ) -> Tuple[Dict[str, Any], float]:
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError(
                "未检测到环境变量 AIROUTER-API-KEY。"
                "如果你的环境使用下划线命名，也支持 AIROUTER_API_KEY。"
            )

        cleaned_base_url = base_url.strip()
        if not cleaned_base_url:
            raise RuntimeError("base_url 不能为空。")
        if "://" not in cleaned_base_url:
            cleaned_base_url = f"https://{cleaned_base_url}"

        url = f"{cleaned_base_url.rstrip('/')}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=(30, timeout_seconds),
        )
        elapsed = time.time() - start_time

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"接口返回了非 JSON 内容，HTTP {response.status_code}，响应片段：{response.text[:300]}"
            ) from exc

        if response.status_code != 200:
            message = response_payload.get("msg") or response_payload.get("message") or str(response_payload)
            hint = _extract_invalid_field_hint(message)
            if hint:
                message = f"{message} ({hint})"
            raise RuntimeError(f"接口请求失败，HTTP {response.status_code}: {message}")

        code = response_payload.get("code")
        if code not in (None, 200, "200"):
            message = response_payload.get("msg") or response_payload.get("message") or str(response_payload)
            raise RuntimeError(f"接口返回异常 code={code}: {message}")

        return response_payload, elapsed

    def _request_gemini_images(
        self,
        payload: Dict[str, Any],
        base_url: str,
        timeout_seconds: int,
        model: str,
    ) -> Tuple[Dict[str, Any], float]:
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError(
                "未检测到环境变量 AIROUTER-API-KEY。"
                "如果你的环境使用下划线命名，也支持 AIROUTER_API_KEY。"
            )

        cleaned_base_url = base_url.strip()
        if not cleaned_base_url:
            raise RuntimeError("base_url 不能为空。")
        if "://" not in cleaned_base_url:
            cleaned_base_url = f"https://{cleaned_base_url}"

        url = f"{cleaned_base_url.rstrip('/')}/v1beta/models/{model.strip()}:generateContent"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=(30, timeout_seconds),
        )
        elapsed = time.time() - start_time

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"接口返回了非 JSON 内容，HTTP {response.status_code}，响应片段：{response.text[:300]}"
            ) from exc

        if response.status_code != 200:
            message = response_payload.get("error", {}).get("message") if isinstance(response_payload, dict) else None
            if not message and isinstance(response_payload, dict):
                message = response_payload.get("msg") or response_payload.get("message") or str(response_payload)
            hint = _extract_invalid_field_hint(message or "")
            if hint:
                message = f"{message} ({hint})"
            raise RuntimeError(f"Gemini 接口请求失败，HTTP {response.status_code}: {message}")

        return response_payload, elapsed

    def _decode_images(
        self,
        data_items: Sequence[Dict[str, Any]],
        timeout_seconds: int,
    ) -> Tuple[List[torch.Tensor], float]:
        tensors: List[torch.Tensor] = []
        decode_started = time.time()

        for item in data_items:
            if not isinstance(item, dict):
                continue

            image_bytes: Optional[bytes] = None

            inline_data_value = _get_nested_inline_data(item)
            if isinstance(inline_data_value, str) and inline_data_value:
                try:
                    image_bytes = _decode_base64_payload(inline_data_value)
                except Exception:
                    image_bytes = None

            base64_value = (
                item.get("base64")
                or item.get("b64_json")
                or item.get("image")
                or item.get("data")
            )
            if isinstance(base64_value, str) and base64_value:
                try:
                    image_bytes = _decode_base64_payload(base64_value)
                except Exception:
                    image_bytes = None

            bytes_value = item.get("bytes")
            if image_bytes is None and isinstance(bytes_value, str) and bytes_value:
                try:
                    image_bytes = _decode_base64_payload(bytes_value)
                except Exception:
                    image_bytes = None

            url_value = item.get("url") or item.get("uri") or item.get("image_url") or item.get("imageUrl")
            if image_bytes is None and isinstance(url_value, str) and url_value:
                response = requests.get(url_value.strip(), timeout=(30, timeout_seconds))
                response.raise_for_status()
                image_bytes = response.content

            if image_bytes is None:
                continue

            image = Image.open(io.BytesIO(image_bytes))
            tensors.append(_pil_to_tensor(image))

        return tensors, time.time() - decode_started

    def _format_log(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: str,
        image_size: str,
        response_format: str,
        seed: int,
        input_image_count: int,
        api_seconds: float,
        decode_seconds: float,
        output_count: int,
        output_resolution: Optional[str],
        texts: Sequence[str],
        response_summary: Optional[Sequence[str]] = None,
        response_body: Optional[str] = None,
        error: Optional[str] = None,
    ) -> str:
        lines = [
            f"{self.NODE_TITLE}",
            f"模型: {model}",
            f"模式: {'图生图' if input_image_count else '文生图'}",
            f"输入图像数: {input_image_count}",
            f"输出图像数: {output_count}",
            f"耗时: API {api_seconds:.2f}s | 解码 {decode_seconds:.2f}s",
        ]

        if error:
            lines.append("")
            lines.append("错误:")
            lines.append(error)

        return "\n".join(lines)

    def generate(
        self,
        prompt: str,
        model: str,
        aspect_ratio: str,
        image_size: str,
        response_format: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        timeout_seconds: int,
        base_url: str,
        seed: int,
        image_1: Optional[torch.Tensor] = None,
        image_2: Optional[torch.Tensor] = None,
        image_3: Optional[torch.Tensor] = None,
        image_4: Optional[torch.Tensor] = None,
        image_5: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str]:
        input_images = [
            image
            for image in (image_1, image_2, image_3, image_4, image_5)
            if image is not None
        ]

        if not prompt.strip() and not input_images:
            log = self._format_log(
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                response_format=response_format,
                seed=seed,
                input_image_count=0,
                api_seconds=0.0,
                decode_seconds=0.0,
                output_count=0,
                output_resolution=None,
                texts=[],
                error="请至少提供提示词或一张输入图片。",
            )
            return _placeholder_image(), log

        validation_error = _validate_request_params(
            model=model,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            base_url=base_url,
            seed=seed,
            input_image_count=len(input_images),
        )
        if validation_error:
            log = self._format_log(
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                response_format=response_format,
                seed=seed,
                input_image_count=len(input_images),
                api_seconds=0.0,
                decode_seconds=0.0,
                output_count=0,
                output_resolution=None,
                texts=[],
                response_summary=None,
                response_body=None,
                error=validation_error,
            )
            return _placeholder_image(), log

        try:
            is_gemini = _is_gemini_model(model)
            if is_gemini:
                payload = self._build_gemini_payload(
                    prompt=prompt,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    seed=seed,
                    input_images=input_images,
                )
                response_payload, api_seconds = self._request_gemini_images(
                    payload=payload,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    model=model,
                )
            else:
                payload = self._build_payload(
                    prompt=prompt,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    response_format=response_format,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    input_images=input_images,
                )
                response_payload, api_seconds = self._request_images(
                    payload=payload,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                )

            data_items = _collect_image_items(response_payload)
            if not is_gemini:
                normalized_data_items = _normalize_data_items(response_payload)
                if normalized_data_items:
                    data_items = normalized_data_items + [item for item in data_items if item not in normalized_data_items]
            tensors, decode_seconds = self._decode_images(
                data_items=data_items,
                timeout_seconds=timeout_seconds,
            )
            texts = _extract_texts(response_payload)

            if not tensors:
                log = self._format_log(
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    response_format=response_format,
                    seed=seed,
                    input_image_count=len(input_images),
                    api_seconds=api_seconds,
                    decode_seconds=decode_seconds,
                    output_count=0,
                    output_resolution=None,
                    texts=texts,
                    error="接口没有返回可解析的图片数据。",
                )
                return _placeholder_image(), log

            batch = torch.cat(tensors, dim=0)
            height, width = batch.shape[1], batch.shape[2]
            log = self._format_log(
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                response_format=response_format,
                seed=seed,
                input_image_count=len(input_images),
                api_seconds=api_seconds,
                decode_seconds=decode_seconds,
                output_count=batch.shape[0],
                output_resolution=f"{width}x{height}",
                texts=texts,
            )
            return batch, log
        except Exception as exc:
            log = self._format_log(
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                response_format=response_format,
                seed=seed,
                input_image_count=len(input_images),
                api_seconds=0.0,
                decode_seconds=0.0,
                output_count=0,
                output_resolution=None,
                texts=[],
                error=str(exc),
            )
            return _placeholder_image(), log


class AIRouterSeedreamImageNode(AIRouterImageBase):
    NODE_TITLE = "AIRouter Seedream"
    DEFAULT_MODEL = "seedream-5.0-lite"


class AIRouterGeminiImageNode(AIRouterImageBase):
    NODE_TITLE = "AIRouter Gemini"
    DEFAULT_MODEL = "gemini-3.1-flash-image-preview-c"


NODE_CLASS_MAPPINGS = {
    "AIRouterSeedreamImageNode": AIRouterSeedreamImageNode,
    "AIRouterGeminiImageNode": AIRouterGeminiImageNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "AIRouterSeedreamImageNode": "AIRouter Seedream Image",
    "AIRouterGeminiImageNode": "AIRouter Gemini Image",
}
