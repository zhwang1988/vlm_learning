# Day 39 · 店铺前台挂件

> 预计 3–4h ｜ 📓 `notebooks/day-39_theme_widget.ipynb` ｜ 💻 本地 · `src/shopify/extensions/chat-widget/`
> 前置：Day 38（App 能安装）

## 今日目标（一句话）

做一个 Theme App Extension（App Block）：店铺前台右下角出现客服入口，支持**上传图片**、调用你的 Agent API、样式跟随主题。

## 一、读（60 min）

材料：
- `docs/11-shopify.md` 第 4 节（Theme App Extension / App Block / Polaris）
- `src/shopify/extensions/chat-widget/blocks/chat_widget.liquid` 的 `{% schema %}`

思考题（先自己想，答案在讲义或代码注释里）：
1. 为什么用 App Block 而不是改主题模板？（提示：店主升级主题时你的改动会不会丢）
2. 脚本用 `async` 加载有什么代价？（提示：加载顺序 + 全局变量）
3. 设计模式下为什么必须显示占位内容？

## 二、写（100 min）

`src/shopify/extensions/chat-widget/`（已给实现）

| 函数 / 文件 | 你要做什么 |
|---|---|
| `chat_widget.liquid` | 挂件的 HTML + `{% schema %}`（店主在主题编辑器里能调样式） |
| `{% schema %}` settings | api_base / title / greeting / position / accent_color / disclaimer |
| 异步加载脚本 | `<script src=... async>` 不阻塞首屏渲染 |
| `design_mode` 占位 | 主题编辑器里显示一个提示块，别报错 |
| 图片上传 | 前台支持传图 → 转 base64 → POST 到 `/widget/chat` |
| 免责声明 | 底部一行「本回复由 AI 生成」—— 合规要求 |

## 三、跑（本地（无需 GPU））

```bash
# 起后端（挂件要调它）
python -m src.shopify.app --dev --port 8080
# 看扩展清单
cat src/shopify/extensions/chat-widget/shopify.extension.toml
# （有 Shopify CLI 时）本地预览主题扩展
shopify app dev
```

期望输出（节选）：
```
扩展清单：
  name        = chat-widget
  type        = theme_app_extension
  blocks      = chat_widget.liquid

店铺前台：
  右下角出现圆形入口按钮（跟随 accent_color）
  点击展开对话框，标题 = 「在线客服」
  支持上传图片 → 图片缩略图显示在输入框上方
  发送后收到 Agent 回复

主题编辑器：
  左侧出现「AI 客服挂件」区块，可拖拽 / 调位置 / 改文案
  设计模式下显示：『AI 客服挂件（预览）』
```

## 四、验收清单

- [ ] 在真实商品页上传一张商品图能收到回复（这是 M7 的关键一步）
- [ ] 挂件样式跟随主题（改 `accent_color` 生效）
- [ ] 主题编辑器里能配置，且**设计模式下不报错**
- [ ] 底部有 AI 免责声明

## 五、容易踩的坑

1. **脚本同步加载** —— 阻塞首屏，Lighthouse 分数直接掉到及格线以下。用 `async` + 自执行函数。
2. **设计模式下报错** —— 主题编辑器会预渲染，此时没有真实顾客。代码要判断 `design_mode` 并早退。
3. 样式硬编码不跟随主题 —— 店主换成深色主题你的挂件就瞎了。用 CSS 变量 + schema 配置。
4. 图片上传没限大小 —— 有人传 20 MB 的图，你的 API 直接挂。前台也要校验。
5. 挂件的 origin 校验没做 —— 别人可以把你的挂件脚本嵌到自己的站上免费用。
6. 忘了免责声明 —— 合规问题，审核会卡。

## 六、打卡

复制 `progress/daily-log.md` 模板到文件末尾，三行起：读了什么 / 写了什么 / 卡在哪。
