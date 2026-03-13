# DeepCloneFinder2

DeepCloneFinder 的新仓库。

一个代码克隆检测工具。致力于利用LLM的能力寻找出Type-4克隆。

## 依赖安装

Python 版本：3.12

CUDA 版本：12.1

```
pip3 install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu121 # 安装 PyTorch
conda install -y -c pytorch -c nvidia -c conda-forge faiss-gpu=1.14.1 # 安装 FAISS
pip install -r requirements.txt # 安装其他依赖
```

## 环境配置

新建文件，命名为 `config.py`，并将 `config.py.example` 中的内容复制到 `config.py` 中。

根据你的环境配置 `config.py`。

```
pytest
```

## 使用方法

```
python extract_functions.py
python embedding.py
python kmeans.py
python init_clone_classes.py
python generate_batch_requests.py
```

## 项目结构

```
```
