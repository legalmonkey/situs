"""Hugging Face Token Manager for Situs.

Provides centralized, secure resolution, validation, and persistent storage of
Hugging Face access tokens for seamless model downloads (especially gated models
like Gemma 3 and Llama 3.1).

Hierarchy:
1. Environment variables: HF_TOKEN, HUGGING_FACE_HUB_TOKEN
2. Standard Hugging Face cache: huggingface_hub.get_token() / ~/.cache/huggingface/token
3. Situs user config: ~/.situs/config.json
"""

import json
import logging
import os
import pathlib
import sys
from typing import Optional, Tuple

logger = logging.getLogger("situs.token_manager")


def get_situs_config_dir() -> pathlib.Path:
    """Returns the ~/.situs directory, creating it if necessary."""
    config_dir = pathlib.Path(os.environ.get("SITUS_CONFIG_DIR") or os.path.expanduser("~/.situs"))
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return config_dir


def get_situs_config_path() -> pathlib.Path:
    """Returns the ~/.situs/config.json path."""
    return get_situs_config_dir() / "config.json"


def _read_situs_config() -> dict:
    """Safely reads ~/.situs/config.json."""
    cfg_file = get_situs_config_path()
    if cfg_file.is_file():
        try:
            return json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug(f"Failed to read Situs config file: {e}")
    return {}


def _write_situs_config(data: dict) -> bool:
    """Safely writes ~/.situs/config.json."""
    cfg_file = get_situs_config_path()
    try:
        get_situs_config_dir().mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"Failed to write Situs config file: {e}")
        return False


def get_hf_token() -> Optional[str]:
    """
    Resolves the Hugging Face token following the strict priority hierarchy:
    1. Environment variables (HF_TOKEN, HUGGING_FACE_HUB_TOKEN)
    2. Hugging Face standard cache (~/.cache/huggingface/token via huggingface_hub.get_token())
    3. Situs persistent config (~/.situs/config.json)
    
    Returns None if no token is configured anywhere.
    """
    # 1. Environment variables
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()

    # 2. Hugging Face Hub standard token (reads ~/.cache/huggingface/token or system store)
    try:
        import huggingface_hub
        hub_token = huggingface_hub.get_token()
        if hub_token and hub_token.strip():
            return hub_token.strip()
    except Exception as e:
        logger.debug(f"huggingface_hub.get_token() lookup failed: {e}")

    # Fallback direct check of ~/.cache/huggingface/token
    hf_cache_token_file = pathlib.Path(os.path.expanduser("~/.cache/huggingface/token"))
    if hf_cache_token_file.is_file():
        try:
            t = hf_cache_token_file.read_text(encoding="utf-8").strip()
            if t:
                return t
        except Exception:
            pass

    # 3. Situs user configuration
    cfg = _read_situs_config()
    s_token = cfg.get("hf_token")
    if s_token and isinstance(s_token, str) and s_token.strip():
        return s_token.strip()

    return None


def get_hf_token_source() -> Optional[str]:
    """Returns a user-friendly description of where the active token was discovered."""
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return f"Environment variable ({env_key})"

    try:
        import huggingface_hub
        hub_token = huggingface_hub.get_token()
        if hub_token and hub_token.strip():
            return "Hugging Face Cache (~/.cache/huggingface/token)"
    except Exception:
        pass

    hf_cache_token_file = pathlib.Path(os.path.expanduser("~/.cache/huggingface/token"))
    if hf_cache_token_file.is_file():
        try:
            if hf_cache_token_file.read_text(encoding="utf-8").strip():
                return "Hugging Face Cache (~/.cache/huggingface/token)"
        except Exception:
            pass

    cfg = _read_situs_config()
    s_token = cfg.get("hf_token")
    if s_token and isinstance(s_token, str) and s_token.strip():
        return "Situs Configuration (~/.situs/config.json)"

    return None


def is_hf_token_configured() -> bool:
    """Returns True if a valid Hugging Face token is found in any location."""
    return get_hf_token() is not None


def mask_token(token: Optional[str]) -> str:
    """Masks a token for secure display (e.g. 'hf_••••••••••••3aB4')."""
    if not token or not isinstance(token, str):
        return "Not configured"
    token = token.strip()
    if len(token) <= 8:
        return "••••••••"
    prefix = token[:3] if token.startswith("hf_") else token[:2]
    suffix = token[-4:]
    masked_middle = "•" * max(8, len(token) - len(prefix) - len(suffix))
    return f"{prefix}{masked_middle}{suffix}"


def validate_hf_token(token: str, timeout: float = 5.0) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validates a Hugging Face token against the Hugging Face Hub API using whoami.
    Returns:
        (is_valid, username, error_message)
    """
    if not token or not isinstance(token, str) or not token.strip():
        return False, None, "Token cannot be empty."

    token = token.strip()
    try:
        import huggingface_hub
        user_info = huggingface_hub.whoami(token=token)
        username = user_info.get("name") or user_info.get("username") or "Authenticated User"
        return True, username, None
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Invalid username or password" in err_msg or "Unauthorized" in err_msg:
            return False, None, "Invalid token: Hugging Face rejected authentication (401 Unauthorized)."
        # Network timeout or offline - allow saving with warning
        return True, None, f"Network verification warning: {err_msg}"


def save_hf_token(token: str, validate: bool = True) -> Tuple[bool, str, Optional[str]]:
    """
    Saves a Hugging Face token to:
    1. Standard Hugging Face cache (~/.cache/huggingface/token)
    2. Situs configuration file (~/.situs/config.json)
    3. Active process environment (os.environ["HF_TOKEN"])
    
    Returns:
        (success: bool, message: str, username: Optional[str])
    """
    if not token or not isinstance(token, str) or not token.strip():
        return False, "Token cannot be empty.", None

    clean_token = token.strip()
    username = None

    if validate:
        valid, user, err = validate_hf_token(clean_token)
        if not valid:
            return False, err or "Token validation failed.", None
        username = user

    # 1. Save via huggingface_hub.login
    try:
        import huggingface_hub
        huggingface_hub.login(token=clean_token, add_to_git_credential=False)
    except Exception as e:
        logger.debug(f"huggingface_hub.login failed: {e}; writing token file directly.")
        try:
            hf_token_path = pathlib.Path(os.path.expanduser("~/.cache/huggingface/token"))
            hf_token_path.parent.mkdir(parents=True, exist_ok=True)
            hf_token_path.write_text(clean_token, encoding="utf-8")
        except Exception as write_err:
            logger.warning(f"Failed to write direct HF token file: {write_err}")

    # 2. Save in ~/.situs/config.json
    cfg = _read_situs_config()
    cfg["hf_token"] = clean_token
    _write_situs_config(cfg)

    # 3. Update current process environment
    os.environ["HF_TOKEN"] = clean_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = clean_token

    user_note = f" (User: '{username}')" if username else ""
    return True, f"Hugging Face token successfully configured and verified{user_note}!", username


def clear_hf_token() -> bool:
    """Removes stored Hugging Face tokens from environment, HF cache, and Situs config."""
    # 1. Clear environment
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    # 2. Clear Hugging Face cache
    try:
        import huggingface_hub
        huggingface_hub.logout()
    except Exception:
        pass

    hf_token_path = pathlib.Path(os.path.expanduser("~/.cache/huggingface/token"))
    if hf_token_path.is_file():
        try:
            hf_token_path.unlink()
        except Exception as e:
            logger.debug(f"Failed to unlink HF token file: {e}")

    # 3. Clear from ~/.situs/config.json
    cfg = _read_situs_config()
    if "hf_token" in cfg:
        cfg.pop("hf_token", None)
        _write_situs_config(cfg)

    return True
