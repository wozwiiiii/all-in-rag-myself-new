"""LLM 生成代码的安全执行沙箱：三层防护。

威胁模型：LLM 生成的代码可能被提示注入操纵，目标是任意文件读写、
命令执行、secret 读取、数据外传。黑名单枚举在原理上无法穷尽
（np.memmap / pd.io.common.os 赋值别名等），因此防护以能力白名单为主：

第一层（静态 AST）：禁止 import/global；禁止下划线属性（挡 dunder 逃逸）；
属性黑名单（to_csv 等导出、文件/网络名）；连续属性链深 ≤2；
名字白名单（pd/np/dfs + 纯函数内建 + 已赋值变量）。

第二层（运行时命名空间）：子进程 globals 中的 pd/np 不是模块本体，
而是 curated 白名单命名空间（无 io/common/os/memmap 等可达路径）；
__builtins__ 同为白名单。即使 AST 层被绕过，可达能力面也是封闭集合。

第三层（进程隔离）：独立子进程执行、超时强杀、输出截断、表数据惰性加载。
已知边界（如实声明）：子进程仍以当前 OS 用户权限运行，若上述两层同时
被绕过，文件系统隔离需依赖 OS 级降权（容器非 root / Job Object），
见 README「安全模型」。
"""

from __future__ import annotations

import ast
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import SandboxViolation

MAX_CODE_CHARS = 8000

FORBIDDEN_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
)

# 属性黑名单：无论挂在哪个对象上都禁止调用/访问（防御纵深，主防线是运行时白名单）
DANGEROUS_ATTRS = {
    # 解释器/系统
    "system", "popen", "fork", "spawn", "kill", "terminate", "exec", "eval",
    "compile", "open", "input", "breakpoint", "exit", "quit",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "__import__",
    # 进程执行族（execve/execv/spawnl 等，防赋值别名触达 os 模块）
    "execve", "execv", "execvp", "execvpe", "execl", "execlp", "spawnl",
    "spawnv", "spawnve", "forkpty", "posix_spawn",
    # pandas/numpy 文件读写原语（任意文件 IO）
    "read_csv", "read_table", "read_fwf", "read_json", "read_html", "read_sql",
    "read_sql_query", "read_sql_table", "read_clipboard", "read_excel",
    "read_hdf", "read_feather", "read_parquet", "read_orc", "read_sas",
    "read_stata", "read_spss", "read_gbq", "read_pickle", "read_xml",
    "fromfile", "frombuffer", "fromstring", "fromregex", "load", "loads",
    "memmap", "loadtxt", "genfromtxt", "save", "savetxt", "savez",
    "savez_compressed",
    # 结果导出（任意文件写入）
    "to_csv", "to_excel", "to_pickle", "to_sql", "to_parquet", "to_hdf",
    "to_feather", "to_stata", "to_clipboard", "to_msgpack", "to_sqlite",
    "to_xml",
    "savefig",
    # 网络（防御纵深：深链规则已挡，单独命中也拒）
    "urlopen", "urllib", "requests", "httpx", "socket",
}

SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "print", "range", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip", "repr", "format",
    # 异常类：LLM 高频书写 try/except，缺失会浪费自愈迭代（只读用途无风险）
    "Exception", "ValueError", "KeyError", "TypeError", "IndexError",
    "RuntimeError", "ZeroDivisionError", "FileNotFoundError", "StopIteration",
}

ROOT_NAMES = {"pd", "np", "dfs"}


@dataclass
class SandboxResult:
    ok: bool
    output: str = ""
    error: str | None = None


def _collect_assigned_names(tree: ast.Module) -> set[str]:
    """收集代码中所有赋值/绑定产生的名字，它们在 Load 上下文中合法。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.MatchAs) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names


def _attr_chain_depth(node: ast.AST) -> int:
    """连续属性链长度：pd.DataFrame -> 1；pd.io.common.os.environ -> 4。"""
    depth = 0
    cur = node
    while isinstance(cur, ast.Attribute):
        depth += 1
        cur = cur.value
    return depth


def validate_code(code: str) -> ast.Module:
    """静态校验，非法即抛 SandboxViolation（附带面向 LLM 的修正提示）。"""
    if not code or not code.strip():
        raise SandboxViolation("代码为空")
    if len(code) > MAX_CODE_CHARS:
        raise SandboxViolation(f"代码超过 {MAX_CODE_CHARS} 字符上限")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        cjk_hint = ""
        if e.msg and "invalid character" in e.msg:
            cjk_hint = "（代码中混入了中文标点（，。；：（）），请全部替换为英文标点；注释以 # 开头）"
        raise SandboxViolation(f"语法错误: {e.msg} (line {e.lineno}){cjk_hint}") from e

    assigned = _collect_assigned_names(tree)
    allowed_names = ROOT_NAMES | SAFE_BUILTINS | assigned

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise SandboxViolation("禁止 import / global / nonlocal 语句，只能使用预置的 pd、np、dfs")
        if isinstance(node, ast.Attribute):
            # 深链防护：pd.io.common.os.environ / pd.io.common.urlopen 等模块内嵌 os/网络
            # 引用只能经 3 层以上属性链触达；正常分析代码至多两级（df['x'].str.contains）
            if _attr_chain_depth(node) > 2:
                raise SandboxViolation(
                    "禁止两层以上的连续属性访问（如 a.b.c.d）：请只用标准 API，"
                    "如 df.groupby(...)、np.linalg.norm(...)、df['col'].str.contains(...)"
                )
            if node.attr.startswith("_"):
                raise SandboxViolation(f"禁止访问下划线属性: .{node.attr}")
            if node.attr in DANGEROUS_ATTRS:
                raise SandboxViolation(f"禁止调用方法: .{node.attr}（不允许读写文件或执行系统命令）")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in allowed_names:
                raise SandboxViolation(
                    f"未定义或不允许使用的名称: {node.id}（可用: pd / np / dfs，或先赋值再使用）"
                )
    return tree


# ---------------------------------------------------------------------------
# 子进程侧：运行时隔离
# ---------------------------------------------------------------------------

import builtins as _builtins_module

_SAFE_BUILTINS_DICT = {
    name: getattr(_builtins_module, name)
    for name in SAFE_BUILTINS
    if hasattr(_builtins_module, name)
}


class _LazyTables(dict):
    """dfs 字典：首次访问某张表时才从 parquet 加载，避免全量加载拖慢执行。"""

    def __init__(self, table_names: list[str], parquet_dir: Path):
        super().__init__()
        self._names = set(table_names)
        self._dir = parquet_dir

    def __contains__(self, key: object) -> bool:
        return key in self._names or super().__contains__(key)

    def __missing__(self, key: str):
        if key in self._names:
            import pandas as pd

            self[key] = pd.read_parquet(self._dir / f"{key}.parquet")
            return self[key]
        raise KeyError(key)


def _render_result(res) -> str:
    import pandas as pd

    if isinstance(res, pd.DataFrame):
        return res.head(30).to_string()
    if isinstance(res, pd.Series):
        return res.head(50).to_string()
    return str(res)


def _build_pd_namespace():
    """curated pandas 命名空间：不暴露模块本体，io/common/os 等模块内引用不可达。"""
    import types

    import pandas as pd

    members = {
        "DataFrame": pd.DataFrame,
        "Series": pd.Series,
        "Index": pd.Index,
        "concat": pd.concat,
        "merge": pd.merge,
        "melt": pd.melt,
        "pivot_table": pd.pivot_table,
        "to_datetime": pd.to_datetime,
        "to_numeric": pd.to_numeric,
        "isna": pd.isna,
        "notna": pd.notna,
        "NA": pd.NA,
        "NaT": pd.NaT,
        "cut": pd.cut,
        "qcut": pd.qcut,
        "date_range": pd.date_range,
    }
    return types.SimpleNamespace(**members)


def _build_np_namespace():
    """curated numpy 命名空间：无 memmap/loadtxt/save 等文件原语。"""
    import types

    import numpy as np

    members = {
        "array": np.array,
        "asarray": np.asarray,
        "nan": np.nan,
        "inf": np.inf,
        "mean": np.mean,
        "sum": np.sum,
        "min": np.min,
        "max": np.max,
        "std": np.std,
        "var": np.var,
        "median": np.median,
        "unique": np.unique,
        "where": np.where,
        "arange": np.arange,
        "linspace": np.linspace,
        "zeros": np.zeros,
        "ones": np.ones,
        "full": np.full,
        "concatenate": np.concatenate,
        "vstack": np.vstack,
        "hstack": np.hstack,
        "isnan": np.isnan,
        "argmax": np.argmax,
        "argmin": np.argmin,
        "abs": np.abs,
        "round": np.round,
        "sort": np.sort,
        "argsort": np.argsort,
        "cumsum": np.cumsum,
        "percentile": np.percentile,
        "quantile": np.quantile,
        # linalg 为浅子模块（norm/inv/det 等），无文件/网络能力
        "linalg": types.SimpleNamespace(
            norm=np.linalg.norm, inv=np.linalg.inv, det=np.linalg.det,
            eig=np.linalg.eig, svd=np.linalg.svd, solve=np.linalg.solve,
        ),
    }
    # numpy 2.x 首次 str(ndarray) 会惰性配置 __str__，内部依赖 __import__；
    # 沙箱 __builtins__ 不含 __import__，故在此处（builtins 仍正常）预触发，
    # 避免用户代码 str(array) 时报 "Unable to configure default ndarray.__str__"。
    try:
        str(np.array([0]))
    except Exception:
        pass
    return types.SimpleNamespace(**members)


def _child_main(conn, code: str, parquet_dir: str, max_output_chars: int) -> None:
    """在独立子进程中运行；任何异常都转为结构化错误回传，绝不让子进程崩溃外泄。"""
    import io
    import sys
    import traceback
    from pathlib import Path

    try:
        pq_dir = Path(parquet_dir)
        table_names = [p.stem for p in pq_dir.glob("*.parquet")]
        dfs = _LazyTables(table_names, pq_dir)

        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            globs: dict = {
                "pd": _build_pd_namespace(),
                "np": _build_np_namespace(),
                "dfs": dfs,
                "__builtins__": dict(_SAFE_BUILTINS_DICT),
            }
            exec(compile(code, "<sandbox>", "exec"), globs)  # noqa: S102 - 三层防护下的受控执行

        out = buf.getvalue()
        if "result" in globs:
            rendered = _render_result(globs["result"])
            out = f"{out}\n[result] {rendered}" if out else f"[result] {rendered}"
        conn.send({"ok": True, "output": out[:max_output_chars], "error": None})
    except BaseException:
        tb = traceback.format_exc()
        conn.send({"ok": False, "output": "", "error": tb[-2000:]})
    finally:
        conn.close()


def run_code(
    code: str,
    *,
    parquet_dir: Path,
    timeout_s: float,
    max_output_chars: int = 4000,
) -> SandboxResult:
    """校验并执行 LLM 生成的 pandas 代码，返回沙箱结果（不含异常抛出，便于回填 ReAct）。"""
    validate_code(code)

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child_main,
        args=(child_conn, code, str(parquet_dir), max_output_chars),
        daemon=True,
    )
    proc.start()
    child_conn.close()
    try:
        if parent_conn.poll(timeout_s):
            payload = parent_conn.recv()
        else:
            proc.terminate()
            proc.join(5)
            return SandboxResult(
                ok=False,
                error=f"TimeoutError: 代码执行超过 {timeout_s}s，已被强制终止。请检查是否存在死循环，或改用更高效的向量化写法。",
            )
    except (EOFError, OSError):
        return SandboxResult(ok=False, error="子进程异常退出，未产生输出")
    finally:
        proc.join(5)
        parent_conn.close()

    return SandboxResult(ok=payload["ok"], output=payload["output"], error=payload["error"])
