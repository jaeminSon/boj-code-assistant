import os
import subprocess
import glob
import weave
from typing import List, Dict

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

import weave
import yaml
import json

def read_yaml(path_yaml: str) -> Dict:
    with open(path_yaml, "r") as file:
        data = yaml.safe_load(file)
    return data

HOME_DIR  = os.path.dirname(os.path.realpath(__file__)).split('models')[0]
AUTH = read_yaml(os.path.join(HOME_DIR, "authentication/api_key.yaml"))
TAGS = json.load(open(os.path.join(HOME_DIR, "metainfo/problem_tag.json")))
PROB_LISTS = json.load(open(os.path.join(HOME_DIR, "metainfo/problem_lists.json")))
VDB = None
PATH_DRAFT = os.path.join(HOME_DIR, "draft.py")

def _vdb_exists(homedir) -> bool:
    return os.path.exists(os.path.join(homedir, "chroma.sqlite3"))

def initialize_vdb(homedir):
    def find_txt_files(directory: str) -> List[str]:
        return glob.glob(os.path.join(directory, "**/*.txt"), recursive=True)

    # save vdb to persistent directory
    docs = [TextLoader(f).load() for f in find_txt_files(homedir)]
    docs_list = [item for sublist in docs for item in sublist]
    Chroma.from_documents(
        documents=docs_list,
        collection_name="rag-chroma",
        embedding=GPT4AllEmbeddings(),
        persist_directory=homedir,
    )

def prepare_vdb() -> None:
    homedir = os.path.join(HOME_DIR, "descriptions")
    if not _vdb_exists(homedir):
        initialize_vdb(homedir)

    # load from chroma.sqlite3 under homedir
    vdb = Chroma(
        collection_name="rag-chroma",
        embedding_function=GPT4AllEmbeddings(),
        persist_directory=homedir,
    )

    return vdb