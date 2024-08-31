# Dragtraffic
The Repo for [Dragtraffic: Interactive and Controllable Traffic Scene Generation for Autonomous Driving](https://chantsss.github.io/Dragtraffic/).

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
# Clone the repository and set up the environment:

git clone https://github.com/chantsss/Dragtraffic
cd dragtraffic

# Create a virtual environment
conda create -n dragtraffic python=3.9.12
conda activate dragtraffic

# Install PyTorch and other dependencies
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116 
pip install -e .
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
