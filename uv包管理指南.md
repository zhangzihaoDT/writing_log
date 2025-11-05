`uv` 是一个现代化、高性能的 Python 包管理器，它兼容 `pip`/`venv`，并使用 `pyproject.toml` 做项目管理。你可以把它理解为“**快得多的 pip + virtualenv + poetry 的组合替代品**”。

下面是最常用的 `uv` 命令，一图一表尽览 👇

---

## 🚀 常用 `uv` 命令速查表

| 功能             | 命令                                   | 说明                                        |
| -------------- | ------------------------------------ | ----------------------------------------- |
| 初始化项目          | `uv init`                            | 生成 `pyproject.toml`，设置虚拟环境                |
| 添加依赖           | `uv add <包名>`                        | 安装包并添加到 `pyproject.toml`                  |
| 移除依赖           | `uv remove <包名>`                     | 卸载包并从项目配置中移除                              |
| 安装全部依赖         | `uv pip install`                     | 从 `pyproject.toml` + `uv.lock` 安装所有依赖     |
| 安装某些依赖（兼容 pip） | `uv pip install -r requirements.txt` | 支持传统 `requirements.txt`                   |
| 冻结依赖（导出）       | `uv pip freeze > requirements.txt`   | 导出当前环境为 `requirements.txt`                |
| 运行脚本（自动启用虚拟环境） | `uv run <命令>`                        | 类似 `poetry run`，如 `uv run python main.py` |
| 激活虚拟环境         | `source .venv/bin/activate`          | 传统方式，Jupyter 等外部环境使用时需要                   |
| 列出已安装包         | `uv pip list`                        | 查看所有已安装依赖                                 |
| 列出主动安装的包       | `uv pip list --user-requested`       | 显示你 `uv add` 添加的主依赖                       |
| 更新依赖版本锁定       | `uv sync`                            | 重新解析并写入 `uv.lock`                         |
| 清理缓存           | `uv cache clean`                     | 删除缓存依赖包                                   |

---

## 🧱 示例项目结构

运行以下命令：

```bash
uv init
uv add requests
uv run python -c "import requests; print(requests.__version__)"
```

你会得到一个典型项目结构：

```
your-project/
├── .venv/
├── pyproject.toml
├── uv.lock
└── main.py
```

---

## ✅ 最推荐使用方式：

### 项目第一次启动：

```bash
uv init
uv add <所需库>
```

### 克隆别人项目后：

```bash
uv pip install  # 自动用 .venv 安装 pyproject.toml 里的依赖
```

---

## 🔍 进阶功能（如有需要我可详细介绍）

* `uv lint`（待支持）：即将集成 `ruff`、`black`
* `uv pip install -e .`：支持本地开发模式安装（开发库调试）
* 环境隔离比 `conda` 更轻量，比 `poetry` 更快，兼容性更好

---

如你希望我给出一套完整的 uv 项目模板（含 `.gitignore`、README、notebook 环境注册等），我可以一键帮你生成。是否需要？
