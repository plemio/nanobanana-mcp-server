import base64
import logging
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

import httpx

from google import genai
from google.genai import types as gx

from ..config.settings import (
    AuthMethod,
    BaseModelConfig,
    FlashImageConfig,
    GeminiConfig,
    NanoBanana2Config,
    ProImageConfig,
    ServerConfig,
)
from ..core.exceptions import AuthenticationError, ImageGenerationBlocked


class GeminiClient:
    """Wrapper for Google Gemini API client with multi-model support."""

    def __init__(
        self,
        config: ServerConfig,
        gemini_config: GeminiConfig | BaseModelConfig | FlashImageConfig | ProImageConfig,
    ):
        self.config = config
        self.gemini_config = gemini_config
        self.logger = logging.getLogger(__name__)
        self._client: genai.Client | None = None
        self._http_client: httpx.Client | None = None
        self._client_lock = Lock()

    @property
    def client(self) -> genai.Client:
        """Lazy initialization of Gemini client."""
        with self._client_lock:
            if self._client is not None and (
                self._http_client is None or not self._http_client.is_closed
            ):
                return self._client

            http_options: gx.HttpOptionsDict = {}
            if self.config.gemini_base_url:
                http_options["base_url"] = self.config.gemini_base_url
                safe_url = self._get_safe_base_url_for_log(self.config.gemini_base_url)
                self.logger.info(f"Using custom base URL: {safe_url}")

            timeout_seconds = getattr(self.gemini_config, "request_timeout", 60)
            http_options["timeout"] = timeout_seconds * 1000
            self.logger.debug(f"Request timeout: {timeout_seconds}s")

            if self.config.auth_method == AuthMethod.API_KEY and not self.config.gemini_api_key:
                raise AuthenticationError("API key is required for API_KEY auth method")

            transport = httpx.Client(follow_redirects=True, timeout=timeout_seconds)
            http_options["httpx_client"] = transport
            try:
                if self.config.auth_method == AuthMethod.API_KEY:
                    client = genai.Client(
                        api_key=self.config.gemini_api_key,
                        http_options=http_options,
                    )
                    auth_description = "API Key (Developer API)"
                else:
                    client = genai.Client(
                        vertexai=True,
                        project=self.config.gcp_project_id,
                        location=self.config.gcp_region,
                        http_options=http_options,
                    )
                    auth_description = f"ADC (Vertex AI - {self.config.gcp_region})"
            except Exception:
                transport.close()
                raise
            self._http_client = transport
            self._client = client
            self._log_auth_method(auth_description)
            return client

    def __del__(self) -> None:
        transport = getattr(self, "_http_client", None)
        if transport is not None:
            transport.close()

    @staticmethod
    def _get_safe_base_url_for_log(raw_url: str) -> str:
        """Return a sanitized base URL for logs (no credentials/query/fragment/path)."""
        parsed = urlsplit(raw_url.strip())
        if parsed.scheme and parsed.hostname:
            host = parsed.hostname
            if parsed.port:
                host = f"{host}:{parsed.port}"
            return f"{parsed.scheme}://{host}"
        return "[invalid-base-url]"

    def _log_auth_method(self, method: str):
        """Log the authentication method in use."""
        self.logger.info(f"Authentication method: {method}")

    def validate_auth(self) -> bool:
        """Validate authentication credentials (optional).

        Note: This makes an API call, so use sparingly.
        """
        try:
            # Lightweight API call
            client = self.client
            _ = client.models.list()
            return True
        except Exception as e:
            self.logger.error(f"Authentication validation failed: {e}")
            return False

    def create_image_parts(self, images_b64: list[str], mime_types: list[str]) -> list[gx.Part]:
        """Convert base64 images to Gemini Part objects."""
        if not images_b64 or not mime_types:
            return []

        if len(images_b64) != len(mime_types):
            raise ValueError(
                f"Images and MIME types count mismatch: {len(images_b64)} vs {len(mime_types)}"
            )

        parts = []
        for i, (b64, mime_type) in enumerate(zip(images_b64, mime_types, strict=False)):
            if not b64 or not mime_type:
                self.logger.warning(f"Skipping empty image or MIME type at index {i}")
                continue

            try:
                raw_data = base64.b64decode(b64)
                if len(raw_data) == 0:
                    self.logger.warning(f"Skipping empty image data at index {i}")
                    continue

                part = gx.Part.from_bytes(data=raw_data, mime_type=mime_type)
                parts.append(part)
            except Exception as e:
                self.logger.error(f"Failed to process image at index {i}: {e}")
                raise ValueError(f"Invalid image data at index {i}: {e}") from e
        return parts

    def generate_content(
        self,
        contents: list,
        config: dict[str, Any] | None = None,
        aspect_ratio: str | None = None,
        **kwargs,
    ) -> any:
        """
        Generate content using Gemini API with model-aware parameter handling.

        Args:
            contents: Content list (text, images, etc.)
            config: Generation configuration dict (model-specific parameters)
            aspect_ratio: Optional aspect ratio string (e.g., "16:9")
            **kwargs: Additional parameters

        Returns:
            API response object
        """
        try:
            # Remove unsupported request_options parameter
            kwargs.pop("request_options", None)

            # Check for config conflict
            config_obj = kwargs.pop("config", None)
            if config_obj is not None:
                if aspect_ratio or config:
                    self.logger.warning(
                        "Custom 'config' kwarg provided; ignoring aspect_ratio and config parameters"
                    )
                kwargs["config"] = config_obj
            else:
                # Filter parameters based on model capabilities
                filtered_config = self._filter_parameters(config or {})

                # Build generation config - use TEXT,IMAGE for Pro model compatibility
                config_kwargs = {
                    "response_modalities": ["TEXT", "IMAGE"],
                }

                # Build ImageConfig with aspect_ratio and image_size
                image_config_kwargs = {}
                if aspect_ratio:
                    image_config_kwargs["aspect_ratio"] = aspect_ratio

                # Map resolution to image_size for Pro model
                resolution = config.get("resolution") if config else None
                if resolution:
                    # Map resolution names to API image_size values
                    resolution_map = {
                        "4k": "4K",
                        "2k": "2K",
                        "1k": "1K",
                        "high": "1K",  # Default high to 1K
                    }
                    image_size = resolution_map.get(resolution.lower(), "1K")
                    image_config_kwargs["image_size"] = image_size
                    self.logger.info(f"Setting image_size={image_size} for resolution={resolution}")

                if image_config_kwargs:
                    config_kwargs["image_config"] = gx.ImageConfig(**image_config_kwargs)

                # Merge filtered config parameters (excluding image_config related ones)
                config_kwargs.update(filtered_config)

                kwargs["config"] = gx.GenerateContentConfig(**config_kwargs)

            # Prepare kwargs
            api_kwargs = {
                "model": self.gemini_config.model_name,
                "contents": contents,
            }

            # Merge additional kwargs
            api_kwargs.update(kwargs)

            self.logger.debug(
                f"Calling Gemini API: model={self.gemini_config.model_name}, "
                f"config={api_kwargs.get('config')}"
            )

            client = self.client
            response = client.models.generate_content(**api_kwargs)
            return response

        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            raise

    def _filter_parameters(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Filter configuration parameters based on model capabilities.

        Ensures we only send parameters that the current model supports,
        preventing API errors from unsupported parameters.

        Args:
            config: Raw configuration dictionary

        Returns:
            Filtered configuration with only supported parameters
        """
        if not config:
            return {}

        filtered = {}

        # Common parameters (supported by all models)
        for param in ["temperature", "top_p", "top_k", "max_output_tokens"]:
            if param in config:
                filtered[param] = config[param]

        # NB2-specific parameters
        if isinstance(self.gemini_config, NanoBanana2Config):
            if "thinking_level" in config:
                filtered["thinking_config"] = gx.ThinkingConfig(
                    thinking_level=gx.ThinkingLevel[config["thinking_level"].upper()]
                )

        # Pro-specific parameters - thinking_level is NOT supported by gemini-3-pro-image-preview
        elif isinstance(self.gemini_config, ProImageConfig):
            # Resolution is handled via ImageConfig.image_size, not here
            # Grounding is controlled via prompt/system instructions
            if "thinking_level" in config:
                self.logger.info(
                    "Note: thinking_level is not supported by gemini-3-pro-image-preview, ignoring"
                )

        else:
            # Flash model - warn if Pro parameters are used
            pro_params = ["thinking_level", "media_resolution", "output_resolution"]
            used_pro_params = [p for p in pro_params if p in config]
            if used_pro_params:
                self.logger.warning(
                    f"Pro-only parameters ignored for Flash model: {used_pro_params}"
                )

        return filtered

    def extract_images(self, response) -> list[bytes]:
        """Extract image bytes from Gemini response."""
        images = []
        candidates = getattr(response, "candidates", None)
        if not candidates or len(candidates) == 0:
            return images

        first_candidate = candidates[0]
        if not hasattr(first_candidate, "content") or not first_candidate.content:
            return images

        # `parts` is a declared field, so it is always *present* — a refused generation sets
        # it to None. `getattr(..., [])` only covers a missing attribute, hence the `or []`.
        content_parts = getattr(first_candidate.content, "parts", None) or []
        for part in content_parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and hasattr(inline_data, "data") and inline_data.data:
                images.append(inline_data.data)

        return images

    def describe_block(self, response) -> str | None:
        """Explain why a response carries no image, or None if it does carry one.

        Gemini signals a refusal with HTTP 200 plus an empty candidate, so the reason only
        lives in `finish_reason` / `prompt_feedback.block_reason`. The returned sentence is
        read by an LLM: it names the code, gives the usual cause, and rules out an identical
        retry — without that last part agents replay the same blocked prompt for several turns.
        """
        if self.extract_images(response):
            return None

        code = None
        detail = None

        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            code = self._enum_name(getattr(prompt_feedback, "block_reason", None))
            detail = getattr(prompt_feedback, "block_reason_message", None)

        candidates = getattr(response, "candidates", None) or []
        if not code and candidates:
            first_candidate = candidates[0]
            code = self._enum_name(getattr(first_candidate, "finish_reason", None))
            detail = detail or getattr(first_candidate, "finish_message", None)
            blocked_ratings = [
                self._enum_name(getattr(rating, "category", None))
                for rating in (getattr(first_candidate, "safety_ratings", None) or [])
                if getattr(rating, "blocked", False)
            ]
            if blocked_ratings:
                detail = detail or ", ".join(filter(None, blocked_ratings))

        code = code or "UNKNOWN"

        message = (
            f"Gemini refused to produce an image (reason: {code}). "
            "Usual causes: a copyright-protected character or brand, the likeness of a real "
            "person, or an unsafe subject. Rewrite the prompt with an original description "
            "instead — retrying the same prompt will be refused again."
        )
        if detail:
            message = f"{message} Gemini said: {detail}"
        return message

    def extract_images_or_raise(self, response) -> list[bytes]:
        """`extract_images`, but a refusal raises instead of yielding a silent empty list.

        Every caller treats "no image" as a failure, and previously each one either crashed
        on the None parts or fell through to a generic "no images were generated" dead end.
        """
        images = self.extract_images(response)
        if images:
            return images

        reason = self.describe_block(response)
        self.logger.warning(f"Gemini returned no image: {reason}")
        raise ImageGenerationBlocked(reason)

    @staticmethod
    def _enum_name(value) -> str | None:
        """Render a google.genai enum (or plain string) as its bare name."""
        if value is None:
            return None
        name = getattr(value, "name", None)
        if name:
            return str(name)
        text = str(value)
        return text.rsplit(".", 1)[-1] if "." in text else text

    def upload_file(self, file_path: str, _display_name: str | None = None):
        """Upload file to Gemini Files API.

        Note: display_name is kept for API compatibility but ignored as the
        Gemini Files API does not support display_name parameter in upload.
        """
        try:
            # Gemini Files API only accepts file parameter
            client = self.client
            return client.files.upload(file=file_path)
        except Exception as e:
            self.logger.error(f"File upload error: {e}")
            raise

    def get_file_metadata(self, file_name: str):
        """Get file metadata from Gemini Files API."""
        try:
            client = self.client
            return client.files.get(name=file_name)
        except Exception as e:
            self.logger.error(f"File metadata error: {e}")
            raise
