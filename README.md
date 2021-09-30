# OKNLP

[![Test Linux](https://github.com/PLNUHT/oknlp/actions/workflows/test_linux.yaml/badge.svg)](https://github.com/PLNUHT/oknlp/actions/workflows/test_linux.yaml)
[![Test Mac OS](https://github.com/PLNUHT/oknlp/actions/workflows/test_macos.yaml/badge.svg)](https://github.com/PLNUHT/oknlp/actions/workflows/test_macos.yaml)
[![Test Windows](https://github.com/PLNUHT/oknlp/actions/workflows/test_windows.yaml/badge.svg)](https://github.com/PLNUHT/oknlp/actions/workflows/test_windows.yaml)
[![PyPI](https://img.shields.io/pypi/v/oknlp)](https://pypi.org/project/oknlp/)
[![Documentation Status](https://readthedocs.org/projects/oknlp/badge/?version=stable)](https://oknlp.readthedocs.io/zh/stable/?badge=stable)
[![codecov](https://codecov.io/gh/PLNUHT/oknlp/branch/main/graph/badge.svg?token=BPKY276BB4)](https://codecov.io/gh/PLNUHT/oknlp)


## 安装方法

### CPU only

```shell
$ pip install "oknlp[cpu]"
```

### GPU

请参考 [安装 - OKNLP文档](https://oknlp.readthedocs.io/zh/stable/installation.html)

## 系统支持

|           | Windows | Linux | Mac OS |
| :-------: | :-----: | :---: | :----: |
| Python3.6 |         |   √   |   √    |
| Python3.7 |    √    |   √   |   √    |
| Python3.8 |    √    |   √   |   √    |
| Python3.9 |    √    |   √   |   √    |

## 快速入门

### 中文分词

```python
import oknlp

if __name__ == "__main__":
    model = oknlp.cws.get_by_name("thulac")
    model([
        "我爱北京天安门"
    ])
    # [['我', '爱', '北京', '天安门']]
```

完整文档请参考 [中文分词 - OKNLP文档](https://oknlp.readthedocs.io/zh/stable/examples/cws.html)

### 命名实体识别

```python
import oknlp

if __name__ == "__main__":
    model = oknlp.ner.get_by_name("bert")
    model([
        "我爱北京天安门"
    ])
    # [[{'type': 'LOC', 'begin': 2, 'end': 4}, {'type': 'LOC', 'begin': 4, 'end': 7}]]
```

完整文档请参考 [命名实体识别 - OKNLP文档](https://oknlp.readthedocs.io/zh/stable/examples/ner.html)

### 词性标注

```python
import oknlp

if __name__ == "__main__":
    model = oknlp.postagging.get_by_name("bert")
    model([
        "我爱北京天安门"
    ])
    # [[('我', 'PN'), ('爱', 'VV'), ('北京', 'NR'), ('天安门', 'NR')]]
```

完整文档请参考 [词性标注 - OKNLP文档](https://oknlp.readthedocs.io/zh/stable/examples/postagging.html)

### 细粒度实体分类

```python
import oknlp

if __name__ == "__main__":
    model = oknlp.typing.get_by_name("bert")
    model([
        ("我爱北京天安门", (2, 4))
    ])
    # [[('location', 0.7169095873832703), ('place', 0.8128180503845215), ('city', 0.6188656687736511), ('country', 0.12475886940956116)]]
```

完整文档请参考 [细粒度实体分类 - OKNLP文档](https://oknlp.readthedocs.io/zh/stable/examples/typing.html)

## 贡献者列表

<a href="https://github.com/a710128">@a710128</a>

<a href="https://github.com/Yumiko-188">@Yumiko-188</a>

<a href="https://github.com/HuShaoRu">@HuShaoRu</a>

## 相关合作

本项目由腾讯 TencentNLP 部门提供技术与数据支持。
