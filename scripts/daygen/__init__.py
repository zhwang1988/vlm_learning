"""每日讲义内容（Day 1–48）。

每天一个 dict，字段契约见 render.py：
    n / slug / title / where / files / prereq / goal
    read / think / write_title / write_rows / write_note
    run / expect / accept / pits / nb

    w0 → Day 1–4    W1 架构解剖（本地起手，云上跑代码）
    w3 → Day 13–18  W3 SFT 训练
    w4 → Day 19–24  W4 评测
    w5 → Day 25–30  W5 对齐与推理
    w6 → Day 31–36  W6 Agent
    w7 → Day 37–42  W7 Shopify
    w8 → Day 43–48  W8 交付

**Day 5–12 不在这个目录里** —— 那 8 天（W2 数据工程）的材料是早期手写的，
直接放在 `days/` 和 `notebooks/` 里，生成器只负责在总表里静态列出它们。

要改某一天的内容：改对应的 w{N}.py，然后重跑：

    python scripts/gen_days.py            # 全部重建
    python scripts/gen_days.py --week 1   # 只重建 W1（Day 1–4）
"""
from __future__ import annotations

from .w0 import DAYS as _W0
from .w3 import DAYS as _W3
from .w4 import DAYS as _W4
from .w5 import DAYS as _W5
from .w6 import DAYS as _W6
from .w7 import DAYS as _W7
from .w8 import DAYS as _W8

ALL_DAYS: list[dict] = _W0 + _W3 + _W4 + _W5 + _W6 + _W7 + _W8

BY_WEEK: dict[int, list[dict]] = {
    1: _W0, 3: _W3, 4: _W4, 5: _W5, 6: _W6, 7: _W7, 8: _W8,
}

__all__ = ["ALL_DAYS", "BY_WEEK"]
