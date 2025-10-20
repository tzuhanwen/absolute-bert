import time
from pathlib import Path
import subprocess
import datetime
import yaml
import tempfile
import os
import hashlib

from .utils import (
    get_git_root,
    get_git_head_commit_hash,
    load_model_dirs_from_pyproject,
    get_registry_yamls,
    get_file_hash, 
    get_git_file_statuses,
    get_clean_file_paths
)


def pick_candidate(candidates: list[dict]) -> Path | None:
    print("\n📦 模型候選清單（未註冊者可選）：\n")
    
    numbered = []
    option_counter = 0
    for candidate in candidates:
        if candidate['registerable']:
            info = f"({candidate['info']})" if candidate['info'] else ""
            print(f"{option_counter+1}. {info} {candidate['path']}")
            numbered.append(candidate)
            option_counter += 1
        else:
            print(f"  ({candidate['info']}) {candidate['path']}")

    if option_counter == 0:
        print("✅ 沒有尚未註冊的模型。")
        return None
    
    idx = input("\n請輸入要註冊的模型編號： ")

    return numbered[int(idx)-1] if idx.isdigit() and int(idx) < option_counter else None


def choose_model_line(existing_lines: list[str]) -> str:
    print("\n📘 選擇 model name：")
    for i, name in enumerate(existing_lines):
        print(f" {i+1}. {name}")
    print(f" {len(existing_lines)+1}. 新增新 model name")
    idx = input("\n請選擇 model line 編號： ")
    
    return (
        existing_lines[int(idx)-1]
        if idx.isdigit() and int(idx)-1 < len(existing_lines)
        else input("請輸入新的 model name： ")
    )


def get_autofilled_dict(candidate: dict[str]):
    commit_hash = get_git_head_commit_hash()
    file_path = candidate["path"]
    file_hash = candidate["file_hash"]

    return {
        "registered_commit": commit_hash,
        "registered_name": file_path,
        "registered_file_hash": file_hash,
        "latest_available_commit": commit_hash,
        "latest_name": file_path,
        "timestamp": datetime.datetime.now().isoformat()
    }
    

def generate_default_yaml(records: list[dict[str]], autofilled_yaml: dict[str]) -> str:
    existing_versions = [record.get("arch_version", "") for record in records]
    data = {
        "arch_version": f"(最後版本: {sorted(existing_versions)[-1]})" if existing_versions else "v0.1.0",
        "notes": "請填寫說明...",
    } | {k: f"<autofilled> {v}" for k, v in autofilled_yaml.items()}
    return yaml.dump(data, sort_keys=False, allow_unicode=True)


def launch_editor_for_yaml(default_content: str) -> str:
    EDITOR = os.environ.get("EDITOR", "vim")
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w+", encoding="utf-8") as tf:
        tf.write(default_content)
        tf.flush()
        subprocess.run([EDITOR, tf.name])
        try:
            with open(tf.name, encoding="utf-8") as result:
                edited = result.read()
        finally:
            for _ in range(10):
                try:
                    os.unlink(tf.name)
                    break
                except PermissionError:
                    time.sleep(0.1)
    return edited


def main():

    ## 把所以可以註冊的 module 集合起來讓使用者選

    model_dirs = load_model_dirs_from_pyproject()
    # output = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"], capture_output=True, text=True).stdout.strip()
    
    candidates = []
    for model_dir in model_dirs:
        abs_registry_dir = get_git_root() / model_dir / "model_registry"

        ## dict[registered_file_name_without_extension, yaml_content]
        yaml_contents = get_registry_yamls(abs_registry_dir)
        registered_file_hashes = set([record["registered_file_hash"] for content in yaml_contents.values() for record in content['records']])
        
        file_statuses = get_git_file_statuses(model_dir)
        dirty_file_paths = set()
        for status in file_statuses:
            updating_status = {"info": ""}
            
            if status["state"] == "untracked":
                updating_status |= {"registerable": False, "info": "untracked"}

            elif status["state"] == "renamed":
                if status["score"] == "R100":
                    file_bytes = subprocess.run(
                        ["git", "show", f"HEAD:{status['orig_path']}"],
                        capture_output=True,
                        check=True
                    ).stdout
                    file_hash = hashlib.sha256(file_bytes).hexdigest()
                    
                    if file_hash in registered_file_hashes:
                        ## 目前沒辦法 register，主要會是 check 時 update record 並 abort commit
                        updating_status |= {"registerable": False, "info": "renamed, registered"}
                    else:
                        updating_status |= {
                            "registerable": True, 
                            "info": "renamed",
                            "file_hash": file_hash,
                        }   
                else:
                    updating_status |= {"registerable": False, "info": "renamed, dirty"}

            elif status["state"] == "unknown":
                updating_status |= {"registerable": False, "info": "unknown"}

            else:
                updating_status |= {"registerable": False, "info": "dirty"}

            candidates.append(
                {
                    "yaml_contents": yaml_contents,
                    "abs_registry_dir": abs_registry_dir,
                    "path": status["path"]
                } | updating_status
            )
            dirty_file_paths.add(status["path"])
    
        
        ## Issue: 如果 depth 太深，覆蓋到另一個 dir_，可能會有 dir_ 的 dirty_file_paths 不是這個 dir
        ## 的 dirty_file_paths。主要是因為上面的 get_git_file_statuses 只能看到一層的結果，也就是
        ## depth=1。
        clean_paths = get_clean_file_paths(model_dir, dirty_file_paths, depth=1)
        
        for clean_path in clean_paths:
            
            updating_status = {"info": ""}

            file_hash = get_file_hash(clean_path)
            if file_hash in registered_file_hashes:
                updating_status |= {"registerable": False, "info": "registered"}
            else:
                updating_status |= {
                    "registerable": True,
                    "file_hash": file_hash
                }

            candidates.append(
                {
                    "yaml_contents": yaml_contents,
                    "abs_registry_dir": abs_registry_dir,
                    "path": clean_path
                } | updating_status
            )

    candidate = pick_candidate(candidates)

    if not candidate:
        return
    

    ## 選擇要註冊的模型 line

    existing_model_lines = list(candidate['yaml_contents'].keys())
    chosen_yaml_name = choose_model_line(existing_model_lines)

    yaml_content = yaml_contents.get(chosen_yaml_name, {"records": []})


    ## 生成並讓使用者修改這次註冊的資訊，append 到原本的 record list

    autofilled_dict = get_autofilled_dict(candidate)
    default_yaml = generate_default_yaml(yaml_content["records"], autofilled_dict)
    edited_yaml = launch_editor_for_yaml(default_yaml)
    final_dict = yaml.safe_load(edited_yaml) | autofilled_dict
    yaml_content["records"].append(final_dict)


    ## 把新的 record list 轉成文字，覆蓋回原本的 yaml 檔
    candidate["abs_registry_dir"].mkdir(parents=True, exist_ok=True)
    out_path = candidate["abs_registry_dir"] / f"{chosen_yaml_name}.yaml"
    
    out_path.write_text(
        yaml.dump(yaml_content, sort_keys=False, allow_unicode=True), 
        encoding="utf-8"
    )
    print(f"\n✅ 已寫入：{out_path}")



if __name__ == "__main__":
    main()