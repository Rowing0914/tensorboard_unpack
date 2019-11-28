import os

PROJECT_NAME = __file__.split("/")[-3]
_path = os.path.abspath(os.curdir)
ROOT_DIR = _path.split(PROJECT_NAME)[0] + PROJECT_NAME
# print(ROOT_DIR)
# ~/Desktop/temp_pana
