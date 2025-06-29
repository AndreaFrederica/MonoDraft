# MonoDraft

**单色草稿纸模板生成工具**

MonoDraft 专注于将扫描件或图片快速转化为单色（单一前景色 + 透明背景）的草稿纸模板，适用于打印、数字标注或绘图打底。

## 功能亮点

* **自动阈值分析**：基于 Otsu 算法自动选取最佳亮度阈值，一键完成背景去除与前景提取。
* **直方图统计**：可定制灰度分区，终端可视化 ASCII 直方图，直观了解亮度分布。
* **闭运算孔洞填充**：消除前景中的小孔洞与断裂，提高模板完整度。
* **多模式支持**：

  * **手动模式**：指定阈值、前景色与结构元大小，自定义精细化处理。
  * **统计模式**：仅输出亮度分布与推荐阈值。
  * **自动模式**：一键完成统计与降噪，生成最终模板。

## 安装依赖

```bash
pip install pillow numpy opencv-python
```

## 使用示例

### 查看帮助

```bash
python mono_draft.py --help
```

### 统计模式

```bash
python mono_draft.py input.png --stats [--bins 32]
```
```pwsh
R:\ via 🐍 v3.13.3 took 8s
❯ python .\草稿纸模板提取器.py .\5.png --stats
C:\Users\qwe17\AppData\Local\Programs\Python\Python313\Lib\site-packages\PIL\Image.py:3442: DecompressionBombWarning: Image size (135836325 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.
  warnings.warn(
亮度直方图 (bins=32):
  0-  6: -                                                  (0)
  7- 14: -                                                  (0)
 15- 22: -                                                  (0)
 23- 30:                                                    (1)
 31- 38:                                                    (20)
 39- 46:                                                    (34)
 47- 54:                                                    (69)
 55- 62:                                                    (95)
 63- 70:                                                    (173)
 71- 78:                                                    (168)
 79- 86:                                                    (196)
 87- 94:                                                    (492)
 95-102:                                                    (5435)
103-110:                                                    (59707)
111-118:                                                    (374462)
119-126:                                                    (1210663)
127-134:                                                    (1973630)
135-142:                                                    (1837825)
143-150:                                                    (1181191)
151-158:                                                    (689609)
159-166:                                                    (442289)
167-174:                                                    (316041)
175-182:                                                    (250160)
183-190:                                                    (221776)
191-198:                                                    (209074)
199-206:                                                    (215601)
207-214:                                                    (231958)
215-222:                                                    (268708)
223-230:                                                    (360108)
231-238:                                                    (582350)
239-246:                                                    (1525073)
247-254: ################################################## (123879417)
亮度范围: 28 ~ 255
Otsu 推荐阈值: 198
示例命令:
  python 草稿纸模板提取器.py .\5.png out.png 198
  或自定义颜色: python 草稿纸模板提取器.py .\5.png out_red.png 198 --fg 255,0,0
```

输出灰度直方图、亮度范围、Otsu 推荐阈值与示例命令。

### 手动模式

```bash
python mono_draft.py input.png output.png 180 --fg 0,0,0 --kernel 5
```

* `180`：亮度阈值，≤180 保留前景。
* `--fg 0,0,0`：前景色 RGB，默认黑色。
* `--kernel 5`：闭运算结构元大小，默认 5。

### 自动模式

```bash
python mono_draft.py input.png --auto [--fg 0,0,0] [--kernel 5] [--bins 32]
```
```pwsh
R:\ via 🐍 v3.13.3 took 8s
❯ python .\草稿纸模板提取器.py .\5.png --auto
C:\Users\qwe17\AppData\Local\Programs\Python\Python313\Lib\site-packages\PIL\Image.py:3442: DecompressionBombWarning: Image size (135836325 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.
  warnings.warn(
亮度直方图 (bins=32):
  0-  6: -                                                  (0)
  7- 14: -                                                  (0)
 15- 22: -                                                  (0)
 23- 30:                                                    (1)
 31- 38:                                                    (20)
 39- 46:                                                    (34)
 47- 54:                                                    (69)
 55- 62:                                                    (95)
 63- 70:                                                    (173)
 71- 78:                                                    (168)
 79- 86:                                                    (196)
 87- 94:                                                    (492)
 95-102:                                                    (5435)
103-110:                                                    (59707)
111-118:                                                    (374462)
119-126:                                                    (1210663)
127-134:                                                    (1973630)
135-142:                                                    (1837825)
143-150:                                                    (1181191)
151-158:                                                    (689609)
159-166:                                                    (442289)
167-174:                                                    (316041)
175-182:                                                    (250160)
183-190:                                                    (221776)
191-198:                                                    (209074)
199-206:                                                    (215601)
207-214:                                                    (231958)
215-222:                                                    (268708)
223-230:                                                    (360108)
231-238:                                                    (582350)
239-246:                                                    (1525073)
247-254: ################################################## (123879417)
亮度范围: 28 ~ 255
Otsu 推荐阈值: 198
Auto 模式：使用阈值 198 进行降噪，输出 5_auto.png
[+] Saved denoised image with closing: 5_auto.png
```

* 自动计算阈值并生成模板，输出文件默认为 `input_auto.png`。

## 参数说明

| 参数          | 说明                                    | 默认           |
| ----------- | ------------------------------------- | ------------ |
| `input`     | 输入图片路径                                | —            |
| `output`    | 输出 PNG 路径，`--stats/--auto` 可省略        | `*_auto.png` |
| `threshold` | 亮度阈值（0-255），前景保留条件                    | —            |
| `--fg`      | 前景颜色 R,G,B（0-255），如 `--fg 255,0,0` 红色 | `0,0,0`      |
| `--kernel`  | 闭运算结构元尺寸，决定孔洞填充范围                     | `5`          |
| `--stats`   | 统计模式：打印直方图与推荐阈值后退出                    | —            |
| `--bins`    | ASCII 直方图分区数                          | `32`         |
| `--auto`    | 自动模式：统计 + 降噪，一键生成模板并退出                | —            |

## 许可证

MIT © 2025
