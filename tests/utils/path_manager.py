import os

def get_project_root():
    current_dir = os.path.dirname(os.path.abspath("__file__"))  # 当前文件路径
    return os.path.abspath(os.path.join(current_dir, "../../"))  # 返回根目录路径

def get_file_path(relative_path):
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)