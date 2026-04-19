import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_user_config_dir() -> Path:
    """Return OS-native user config directory."""
    # check for environment override (mainly for tests)
    env_dir = os.environ.get('PATTERN_NURSERY_CONFIG_DIR')
    if env_dir:
        return Path(env_dir)

    if sys.platform == 'win32':
        base = os.environ.get('APPDATA')
        if not base:
            base = str(Path.home() / 'AppData' / 'Roaming')
        return Path(base) / 'pattern-nursery'
    elif sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'pattern-nursery'
    else:
        # linux / other
        base = os.environ.get('XDG_CONFIG_HOME')
        if not base:
            base = str(Path.home() / '.config')
        return Path(base) / 'pattern-nursery'


def get_profiles_path() -> Path:
    """Return path to profiles.json."""
    return get_user_config_dir() / 'profiles.json'


def load_profiles() -> dict:
    """Load profiles from profiles.json."""
    path = get_profiles_path()
    if not path.exists():
        return {'version': 1, 'profiles': {}}

    try:
        with open(path, 'r', encoding='utf8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # back up bad file and return empty
        if path.exists():
            bad_path = path.with_suffix('.json.bad')
            try:
                path.replace(bad_path)
            except IOError:
                pass
        return {'version': 1, 'profiles': {}}


def save_profiles(data: dict) -> None:
    """Save profiles to profiles.json atomically."""
    path = get_profiles_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix('.json.tmp')
    with open(tmp_path, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=2)

    tmp_path.replace(path)


def get_profile(profile_name: str) -> dict:
    """Get defaults for a specific profile."""
    data = load_profiles()
    profiles = data.get('profiles', {})
    return profiles.get(profile_name, {})


def set_profile(profile_name: str, work_dir: Path, run_name: str) -> None:
    """Set defaults for a specific profile."""
    data = load_profiles()
    if 'profiles' not in data:
        data['profiles'] = {}

    data['profiles'][profile_name] = {
        'work_dir': str(work_dir),
        'run_name': run_name,
        'updated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }
    save_profiles(data)
