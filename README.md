# DeepCloneFinder2

DeepCloneFinder 的新仓库。

一个代码克隆检测工具。致力于利用LLM的能力寻找出Type-4克隆。

## 依赖安装

Python 版本：3.12

CUDA 版本：12.1

```
pip3 install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu121 # 安装 PyTorch
conda install -c pytorch -c nvidia -c conda-forge faiss-gpu=1.14.1 # 安装 FAISS
pip install -r requirements.txt # 安装其他依赖
```

## 框架配置

新建文件，命名为 `config.py`，并将 `config.py.example` 中的内容复制到 `config.py` 中。

根据你的环境配置 `config.py`。

```
pytest
```

## 项目结构

```
```
