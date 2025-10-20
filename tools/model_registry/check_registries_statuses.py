from pathlib import Path
import subprocess
import yaml
import hashlib
import sys
from typing import Callable

from .utils import (
    get_git_root,
    get_git_head_commit_hash,
    get_file_hash, 
    load_model_dirs_from_pyproject, 
    get_git_file_statuses,
    get_clean_file_paths,
    get_registry_yamls,
    get_git_head_file_content
)


def confirm(prompt: str) -> bool:
    try:
        return input(prompt + " [y/n]: ").strip().lower() == "y"
    except EOFError:
        return False


def handle_updatable_records(record_metadatas: list[dict], update_func: Callable[[dict], None]) -> bool:
    to_abort = False

    if not record_metadatas:
        print("✅ 沒有需要更新的模型記錄。")
        return to_abort

    print(f"📝 有 {len(record_metadatas)} 筆記錄可更新：")
    for metadata in record_metadatas:
        record_info = f"• {metadata['yaml_name']} {metadata['record']['latest_name']}"
        if "new_path" in metadata:
            record_info += f" -> {metadata['new_path']}"
        print(record_info)

    if confirm("要更新所有記錄嗎？ y: 更新，abort, n: 忽略"):
        to_abort = True
        for metadata in record_metadatas:
            update_func(metadata)
        print("✅ 所有記錄已更新，請確認是否需要 stage records 後，再次 commit。")

    return to_abort

def main():
    aborting = False
    git_root = get_git_root()
    head_commit_hash = get_git_head_commit_hash()
    model_dirs = load_model_dirs_from_pyproject()

    register_required = []
    updatable = []
    dir2yamls = {}
    for model_dir in model_dirs:
        abs_registry_dir = git_root / model_dir / "model_registry"
        yaml_contents = get_registry_yamls(abs_registry_dir)
        dir2yamls[model_dir] = yaml_contents
        ## 用法: record_index[file_hash]["record"], record_index[file_hash]["position"]
        record_index = {
            record["registered_file_hash"]: {
                "record": record,
                "model_dir": model_dir,
                "yaml_name": yaml_name, 
                "record_index": idx
            }
            for yaml_name, content in yaml_contents.items()
            for idx, record in enumerate(content['records'])
        }

        file_statuses = get_git_file_statuses(model_dir)
        for status in file_statuses:
            if status["state"] == "untracked":
                continue
            
            elif status["state"] == "renamed":
                if status["score"] == "R100":
                    file_bytes = subprocess.run(
                        ["git", "show", f"HEAD:{status['orig_path']}"],
                        capture_output=True,
                        check=True
                    ).stdout
                    file_hash = hashlib.sha256(file_bytes).hexdigest()

                    if file_hash in record_index:
                        record = record_index[file_hash]["record"]
                        updated_name = record["latest_name"] == status["path"]
                        updated_commit = record["latest_available_commit"] == head_commit_hash
                        if  not updated_name or not updated_commit: 
                            updatable.append(record_index[file_hash] | {"new_path": status["path"]})

                        continue
                
                file_hash = hashlib.sha256(get_git_head_file_content(status["orig_path"])).hexdigest()
                if file_hash not in record_index:
                    register_required.append(model_dir)

                continue
            
            ## modified, deleted, staged, ...
            elif not status["state"] == "added":
                file_hash = hashlib.sha256(get_git_head_file_content(status["path"])).hexdigest()
                if file_hash not in record_index:
                    register_required.append(model_dir)
            
        dirty_file_paths = set(status["state"] for status in file_statuses)
        clean_paths = get_clean_file_paths(model_dir, dirty_file_paths, depth=1)
        for clean_path in clean_paths:
            file_hash = get_file_hash(clean_path)

            if file_hash in record_index:
                record = record_index[file_hash]["record"]
                if not record["latest_available_commit"] == head_commit_hash:
                    updatable.append(record_index[file_hash] | {"path": clean_path})
            else:
                register_required.append(clean_path)
    

    yamls_to_overwrite = set()
    
    def update_and_queue(metadata):
        record = metadata["record"]
        record["latest_available_commit"] = head_commit_hash
        record["latest_name"] = metadata["new_path"] if "new_path" in metadata else metadata["path"]

        yamls_to_overwrite.add((metadata["model_dir"], metadata["yaml_name"]))
    to_abort = handle_updatable_records(updatable, update_and_queue)
    aborting = aborting or to_abort

    for model_dir, yaml_name in yamls_to_overwrite:
        (git_root/model_dir/"model_registry"/f"{yaml_name}.yaml").write_text(
            yaml.dump(dir2yamls[model_dir][yaml_name], sort_keys=False, allow_unicode=True), 
            encoding="utf-8"
        )


    


    if register_required:
        print("❌ The following model modules are staged but not registered:\n")
        for f in register_required:
            print(" -", f)
        print("\nPlease run `register_model.py` before committing.")

        aborting = True

    if aborting:
        sys.exit(1)



if __name__ == "__main__":
    main()