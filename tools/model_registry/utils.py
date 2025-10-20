import os
import subprocess
import yaml
import toml
from pathlib import Path
import hashlib
from typing import Iterator


def get_git_root() -> Path:
    return Path(
        subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        .stdout.strip()
    )


def get_git_head_commit_hash() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def get_git_head_file_content(file_path: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"HEAD:{file_path}"],
        capture_output=True,
        text=False,
        check=True,
    )
    return result.stdout


def load_model_dirs_from_pyproject(pyproject_path: Path | None = None) -> list[Path]:
    if pyproject_path is None:
        pyproject_path = get_git_root() / "pyproject.toml"
    pyproject_data = toml.load(pyproject_path)
    raw_dirs = (
        pyproject_data.get("tool", {})
        .get("model_registry", {})
        .get("model_dirs", [])
    )
    
    return [Path(p) for p in raw_dirs]


def get_registry_yamls(registry_dir: Path) -> dict[str, dict]:
    """
    return: dict[yaml_file_name_without_extension, yaml_content]
    """
    records = {}
    if not registry_dir.exists():
        return records
    for p in registry_dir.glob("*.yaml"):
        try:
            filename_without_extension = os.path.splitext(p.name)[0]
            records[filename_without_extension] = yaml.safe_load(p.read_text(encoding='utf-8'))
        except Exception as e:
            print(e)
            continue
    return records


def get_file_hash(path: Path) -> str:
    path = get_git_root() / path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_clean_file_paths(target_dir: str, dirty_file_paths: set[str], depth=0) -> Iterator[str]:
    """
    depth: 要找尋的深度。
    """

    abs_dir = get_git_root() / target_dir
    file_paths = subprocess.run(
        ["git", "ls-files", "--full-name", abs_dir], capture_output=True, text=True
    ).stdout.splitlines()
    
    is_not_too_deep = lambda path: Path(path).relative_to(target_dir).as_posix().count("/") < depth
    file_paths = filter(is_not_too_deep, file_paths)
    
    clean_paths = filter(lambda name: name not in dirty_file_paths, file_paths)
    
    return clean_paths


# def get_staged_files() -> list[Path]:
#     output = subprocess.run(
#         ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
#         capture_output=True,
#         text=True,
#     ).stdout.strip()
#     return [Path(p) for p in output.splitlines() if p.endswith(".py")]


def safe_split_entries(raw_bytes: bytes) -> list[bytes]:
    """
    安全地切割 Git porcelain -z 的輸出：
    - 用 b'\x00' 切割
    - 自動移除頭尾多餘的 null byte
    - 保證不回傳空行
    """
    stripped = raw_bytes.strip(b'\x00')
    if not stripped:
        return []
    return stripped.split(b'\x00')


def get_git_file_statuses(target_dir: Path) -> list[dict]:
    """
    return:
        list[dict[
            "path": file path relative to git root
        ]]
    """


    ## 不能直接在這裡指定某個 dir 而是指用 git roo dir，是因為如果 renamed file 的 source dir
    ## 跟 target dir 不都在指定的 scope 之中，那會有 file status 從 renamed 變成 added 的問題
    ## 所以在後面才會去篩哪些是在 target_dir 裡面的
    result = subprocess.run(
        ["git", "status", "--porcelain=v2", "-z", "--untracked-files=all", get_git_root()],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        check=True,
    )

    entries = safe_split_entries(result.stdout)
    parsed = []
    i = 0
    
    while i < len(entries):
        
        line = entries[i]

        if line.startswith(b'1 '):
            parts = line.split(b' ', 8)
            if len(parts) < 9:
                raise ValueError(f"Incomplete '1 ' entry at index {i}: {line}")
            _, xy, _, _, _, _, _, _, path_bytes = parts
            path = path_bytes.decode('utf-8', errors='replace')
            
            i += 1
            if not path.startswith(target_dir.as_posix()):
                continue

            x, y = xy.decode()
            
            if x == "A":
                state = "added"
            elif x == "D" or y == "D":
                state = "deleted"
            elif x == "M":
                state = "staged"
            elif y == "M":
                state = "modified"
            elif x == "." and y == ".":
                state = "clean"
            else:
                state = "unknown"

            parsed.append({
                "state": state,
                "path": path,
            })

        elif line.startswith(b'2 '):
            if i + 1 >= len(entries):
                raise ValueError(f"Incomplete '2 ' (rename/copy) entry at index {i}")
            parts = line.split(b' ', 9)
            if len(parts) < 10:
                raise ValueError(f"Incomplete '2 ' entry fields at index {i}: {line}")
            _, xy, _, _, _, _, _, _, score_bytes, path_bytes = parts
            path = path_bytes.decode('utf-8', errors='replace')
            orig_path = entries[i + 1].decode('utf-8', errors='replace')

            i += 2
            if not path.startswith(target_dir.as_posix()):
                continue

            x, y = xy.decode()
            if x == "R":
                state = "renamed"
            elif x == "C":
                state = "copied"
            else:
                state = "unknown"

            parsed.append({
                "state": state,
                "score": score_bytes.decode(),
                "orig_path": orig_path,
                "path": path,
            })
            
        elif line.startswith(b'? '):
            path = line[2:].decode('utf-8', errors='replace')
            
            i += 1
            if not path.startswith(target_dir.as_posix()):
                continue

            parsed.append({
                "state": "untracked",
                "path": path,
            })
            
        elif line.startswith(b'! '):
            path = line[2:].decode('utf-8', errors='replace')
            
            i += 1
            if not path.startswith(target_dir.as_posix()):
                continue

            parsed.append({
                "state": "ignored",
                "path": path,
            })
            
        else:
            i += 1
            parsed.append({
                "state": "unknown",
                "raw": line.decode('utf-8', errors='replace')
            })

    return parsed
