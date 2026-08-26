"""配置测试：多 key 解析、认证开关、默认路径。"""

from app.core.config import Settings


def test_api_key_list_parses_and_strips():
    s = Settings(_env_file=None, api_keys=" k1 , k2 ,, k3 ")
    assert s.api_key_list == ["k1", "k2", "k3"]
    assert s.auth_enabled is True


def test_empty_keys_disables_auth():
    s = Settings(_env_file=None, api_keys="")
    assert s.api_key_list == []
    assert s.auth_enabled is False


def test_defaults_are_deterministic():
    s1 = Settings(_env_file=None)
    s2 = Settings(_env_file=None)
    assert s1.chroma_dir == s2.chroma_dir
    assert s1.top_k == 6
    assert 0.0 < s1.score_threshold < 1.0
