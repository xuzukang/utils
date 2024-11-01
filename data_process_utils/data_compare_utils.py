import numpy as np
import torch

from typing import Callable, Any, Optional, Union, List, Dict, Tuple
from collections import OrderedDict

from rich.table import Table
from rich.text import Text
from rich import print
from rich.console import Console

def log_table(rich_table):
    """Generate an ascii formatted presentation of a Rich table
    Eliminates any column styling
    """
    console = Console(width=150)
    with console.capture() as capture:
        console.print(rich_table)
    return Text.from_ansi(capture.get())


def table_info(
    diff_out: dict,
    table_name: str,
    use_order=False,
    sensitive: str = "cos_similarity",
    reverse: bool = False,
):
    table = Table(title=table_name)

    table.add_column("node output", width=50)
    table.add_column("op_type")
    for key in SENSITIVES:
        table.add_column(key)

    if use_order:
        diff_out = OrderedDict(
            sorted(
                diff_out.items(), key=lambda item: item[1][sensitive], reverse=reverse
            )
        )

    def add_row(
        table: Table,
        key,
        index,
        diff_data,
        is_better: Callable[[Any, Any], Any],
        postprocess: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if postprocess is None:
            postprocess = str
        diff_list = list()
        if is_better(index):
            for sens in SENSITIVES:
                diff_list.append(Text(postprocess(diff_data[sens]), style="bold red"))
        else:
            for sens in SENSITIVES:
                diff_list.append(postprocess(diff_data[sens]))
        table.add_row(key, postprocess(diff_data["op_type"]), *diff_list)

    index = 0
    for key in diff_out.keys():
        key: str
        key_name = key.replace("up_blocks", "up")
        key_name = key_name.replace("attentions", "att")
        key_name = key_name.replace("transformer_blocks", "tb")
        key_name = key_name.replace("resnets", "resn")

        key_name = key_name.replace("down_blocks", "down")
        key_name = key_name.replace("mid_block", "mid")

        key_name = key_name.replace("quantize", "q")

        add_row(table, key_name, index, diff_out[key], lambda index: index < 20)
        index += 1
    # print(table)
    print(f"\n {log_table(table)}")


def print_diff_info(diff_out: dict):
    """
    --------------------------------------------------------
    |             | a_name model | b_name model | abs max diff
    --------------------------------------------------------
    | ****        | ****           | ****             |
    --------------------------------------------------------
    |  op         | ****           | ****             |
    --------------------------------------------------------
    """
    table = Table(title="diff table")
    table.add_column("node output")
    table.add_column("node_type")
    table.add_column("abs mean diff")
    table.add_column("abs mean re diff")
    table.add_column("信噪比snr")

    def add_row(
        table: Table,
        key,
        diff_data,
        is_better: Callable[[Any, Any], Any],
        postprocess: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if postprocess is None:
            postprocess = str
        if is_better(diff_data["mean_ab_diff"]):
            table.add_row(
                key,
                postprocess(diff_data["node_type"]),
                postprocess(diff_data["mean_ab_diff"]),
                postprocess(diff_data["mean_re_diff"]),
                postprocess(diff_data["snr"]),
            )
        # else:
        #     table.add_row(key, postprocess(diff_data["node_type"]),
        #         postprocess(diff_data["max_ab_diff"]), postprocess(diff_data["mean_ab_diff"]), postprocess(diff_data["max_re_diff"]))

    for key in diff_out.keys():
        add_row(table, key, diff_out[key], lambda diff: diff >= 1e-7)

    print(f"\n {log_table(table)}")


def cos_similarity(ta, tb):
    # 计算向量的余弦相似度
    if np.sum(ta * tb) == 0:
        return 0.0
    return np.sum(ta * tb) / (
        np.sqrt(np.sum(np.square(ta))) * np.sqrt(np.sum(np.square(tb)))
    )


def snr(ta, tb):
    # 计算信噪比
    if np.sum(ta**2) == 0:
        return 0.0
    return np.sum((ta - tb) ** 2) / np.sum(ta**2)


def relative_error(ta, tb):
    # 计算相对误差
    if np.mean(np.abs(ta)) == 0:
        return 0.0
    return np.mean(np.abs(ta - tb)) / np.mean(np.abs(ta))


def abs_error(ta, tb):
    # 计算相对误差
    return np.mean(np.abs(ta - tb))


# 把以上三个计算误差的函数放到一个字典中，方便后续调用
Sensitive2Function = {
    "cos_similarity": cos_similarity,
    "snr": snr,
    "relative_error": relative_error,
    "abs_error": abs_error,
}

# 定义敏感度指标
SENSITIVES = ["cos_similarity", "snr", "relative_error", "abs_error"]
