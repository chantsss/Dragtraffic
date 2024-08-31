# Dragtraffic
The Repo for [Dragtraffic: A Non-Expert Interactive and Point-Based Controllable Traffic Scene Generation Framework](https://chantsss.github.io/Dragtraffic/).

**News**: Dragtraffic has been accepted by [**IROS2024**](https://iros2024-abudhabi.org), I will try my best to release the code before the end of august.

[**Webpage**](https://chantsss.github.io/Dragtraffic/) | 
[**Paper**](https://arxiv.org/abs/2404.12624) |


![](assets/overview.jpg)

# We are preparing
We are sorting out the code and the relevant code will be released soon in the following order:
1. - [x] Paper releasing(Accepted).
2. - [x] Model files and checkpoints. 
3. - [ ] UI and Inference demo code.
4. - [ ] Training code.

## Setup environment

```bash
# Clone the code to local
git clone https://github.com/metadriverse/trafficgen.git
cd trafficgen

# Create virtual environment
conda create -n trafficgen python=3.9.12
conda activate trafficgen

# You should install pytorch by yourself to make them compatible with your GPU
# For cuda 11.0:
# pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116 -i https://pypi.tuna.tsinghua.edu.cn/simple
# Install basic dependency
# cd ..
pip install pytorch-lightning -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python
pip install waymo-open-dataset-tf-2-11-0==1.6.0 # for data preprocess
pip install ray -i https://pypi.tuna.tsinghua.edu.cn/simple
```


# Citation
If Dragtraffic is helpful in your research, please consider giving us a star or citing our work:

```bibtex
@misc{wang2024dragtraffic,
      title={Dragtraffic: A Non-Expert Interactive and Point-Based Controllable Traffic Scene Generation Framework}, 
      author={Sheng Wang and Ge Sun and Fulong Ma and Tianshuai Hu and Yongkang Song and Lei Zhu and Ming Liu},
      year={2024},
      eprint={2404.12624},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
