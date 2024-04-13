import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def path_sorting_fn(path):
    path = path.replace("cache_trace_", "").replace("ftds_trace_", "").replace("trace_", "")

    return tuple(map(int, path.split("_")))


def get_volume_paths(volume_dir):
    return sorted(
        glob.glob(f"{volume_dir}/cache_trace*"), key=lambda p: path_sorting_fn(p.split("/")[-1])
    )


def get_test_set_volumes(test_set):
    s = [x for x in glob.glob(f"../trace/Trace/{test_set}/*") if "trace" in x.split("/")[-1]]

    if "scenario_test_trace_simple" in test_set:
        s = s[:1]  # This is big enough set

    if test_set == "1":
        s = s[:1]  # This is big enough set

    return sorted(s, key=lambda p: path_sorting_fn(p.split("/")[-1]))


def read_trace(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="|", skipinitialspace=True)
    df = df.rename(columns=lambda c: c.strip())

    if "print data" in df.columns:
        df.columns = list(df.columns[1:]) + ["temp"]
        df = df.drop(columns=["temp"])
    else:
        df = df.drop(columns=df.columns[0])
        df = df.drop(columns=filter(lambda x: "Unnamed" in x, df.columns))
    df = df.map(lambda x: x.strip() if type(x) == str else x)

    return df


def get_reads(df) -> np.ndarray:
    return df[df["tp"] == "READ"][["objLba", "length"]].values


def get_test_set() -> Dict[str, List[str]]:
    test_dirs = [str(i) for i in range(1, 12)] + ["scenario_test_trace_simple/VDI_virus_scan"]

    return {
        test_dir: [get_volume_paths(v_path) for v_path in get_test_set_volumes(test_dir)]
        for test_dir in test_dirs
    }


def get_page_requests(read_trace: List[Tuple[int, int]], page_size=16 * 1024) -> List[int]:
    requests = []

    for addr, length in read_trace:
        while length > 0:
            requests.append((addr//page_size))
            addr += page_size
            length -= page_size

    return requests
