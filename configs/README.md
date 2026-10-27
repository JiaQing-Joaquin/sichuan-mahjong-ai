# 配置目录

当前规则配置由代码中的 dataclass 提供：

- 对局规则：`GameConfig`（[`src/sichuan_mahjong/game.py`](../src/sichuan_mahjong/game.py)）
  - `swap_three` / `swap_direction`
  - `must_discard_missing`
  - `gang_payment`（刮风下雨）
  - `draw_penalty`（流局）
- 番型：`FanConfig` / `DEFAULT_FAN`（[`src/sichuan_mahjong/scoring.py`](../src/sichuan_mahjong/scoring.py)）

后续把「按平台/地区差异」的规则外置为 JSON/YAML 时，文件放在本目录，
例如 `configs/rules_chengdu.json`、`configs/rules_chongqing.json`。
