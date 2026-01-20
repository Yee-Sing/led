# led

# LED: A Large-Scale Real-World Paired Dataset for Event Camera Denoising

Hi everyone！This is the official repository for the CVPR 2024 paper: **"LED: A Large-Scale Real-World Paired Dataset for Event Camera Denoising"**.

> **Note**: We apologize for the delay in release. Due to a hard drive failure on our original data server, we have spent significant effort re-organizing the data. 

## 🗂️ Download the Dataset

We are releasing the **LED** dataset for your event denoising method training and evaluation.  It contains 600 sequences and 66 sequences in the training/testing part, respectively. Each slice `.npy` file (format: `4 x n`; rows: `x, y, p, t`) represents an event stream spanning **10 ms**.

### The Reference GT Visual Illustration
<p align="center">
  <table align="center">
    <tr>
      <td align="center"><strong>Raw Events</strong></td>
      <td align="center"><strong>Denoised Events</strong></td>
      <td align="center"><strong>Noise Events</strong></td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Yee-Sing/led/main/assets/raw.gif" alt="Raw Events" width="250"/>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Yee-Sing/led/main/assets/denoised.gif" alt="Denoised Events" width="250"/>
      </td>
      <td align="center">
        <img src="https://raw.githubusercontent.com/Yee-Sing/led/main/assets/noise.gif" alt="Noise Events" width="250"/>
      </td>
    </tr>
    <!-- 新增加的一行：跨列的描述单元格 -->
    <tr>
      <td colspan="3" align="center" style="padding-top: 15px; font-style: italic;">
        The nighttime driving scenario
      </td>
    </tr>
  </table>
</p>

The dataset includes:
- **Raw event streams**
- **Denoised GT reference event streams**
- **Separated noise reference event streams**

**Download link:** [LED Dataset](https://pan.baidu.com/s/1ldFgt089GsgLIJh3I57wsQ)
with **Extraction code:** `xwng`

## 🚀 Dataset Usage
1.  Download and extract the `LED` dataset using the link above.
2.  Each `.npy` file can be loaded as follows (example using NumPy):
    ```python
    import numpy as np
    # Load an event slice
    # Data format: 4 rows x n columns(x-integer; y-integer; p-integer; t-integer)
    events = np.load('path/to/xxxxx.npy')
    ```

## ⚙️ Download the Pretrained Model
Our pretrained DTSNN model can be downloaded at [DTSNN](https://pan.baidu.com/s/1W9gDwb7E50aPkfnKp2c-iQ) with the extraction code"5wt7", which is fully implemented on the **Spikingjelly** SNN platform. So, the necessary environment implementation can refer to the guidance from **[Spikingkelly](https://github.com/fangwei123456/spikingjelly)**.

## 🚀 Code Usage
1.  Download and extract the `DTSNN` pretrained model using the link above.
2.  Place the pretrained model in the "./model_zoo/pretrain/SW" directory.
4.  Place the dataset to be tested under the "./data" directory and organize the files according to the corresponding folder hierarchy. Notably, for the convenience of metric calculation, please organize the gt reference into H×W×2 format and place it in the corresponding label directory.
5.  Run the main_tester.py, and the predicted results will be in the "./results" directory.

## 🔄 Future Updates
We will be uploading:
- The official **DTSNN training code**

## Acknowledgement
We would also like to extend our special thanks to other collaborators (Shihan Peng, Hanyu Zhou, Haoyue Liu, Lin Zhu, Wei Zhang, Yi Chang*, Sheng Zhong, Luxin Yan) for their valuable support of this work.

## 📜 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{duan2024led,
  title={LED: A Large-Scale Real-World Paired Dataset for Event Camera Denoising},
  author={Duan, Yuxing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={25637--25647},
  year={2024}
}
