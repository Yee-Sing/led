# led

# LED: A Large-Scale Real-World Paired Dataset for Event Camera Denoising

Hi everyone！This is official repository for the CVPR 2024 paper: **"LED: A Large-Scale Real-World Paired Dataset for Event Camera Denoising"**.

> **Note**: We apologize for the delay in release. Due to a hard drive failure on our original data server, we have spent significant effort re-organizing the data. We are now releasing the **LED_Test** dataset to facilitate performance benchmarking.

## 🗂️ Download the Dataset

We are releasing the **LED_Test** dataset for your event denoising method evaluation. It contains **66 sequences**. Each slice `.npy` file (format: `4 x n`; rows: `x, y, p, t`) represents an event stream spanning **10 ms**.

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
  </table>
</p>

The dataset includes:
- **Raw event streams**
- **Denoised GT reference event streams**
- **Separated noise event streams**

**Download link:** [Baidu Netdisk](https://pan.baidu.com/s/1Z7OU7HFcji7bwRZFLLU6bg?pwd=3s14)
**Extraction code:** `3s14`

## 🚀 Usage

1.  Download and extract the `LED_Test` dataset using the link above.
2.  Each `.npy` file can be loaded as follows (example using NumPy):
    ```python
    import numpy as np
    # Load an event slice
    # Data format: 4 rows x n columns
    # Row 0: x (pixel coordinate, integer)
    # Row 1: y (pixel coordinate, integer)
    # Row 2: p (polarity, integer, typically 1(on event) or 0(off event))
    # Row 3: t (timestamp, integer, in microsecond)
    events = np.load('path/to/xxxxx.npy')
    ```

## 🔄 Future Updates

We will be uploading:
- **More training data** in phases
- The official **DTSNN Pretrain model and relevant code**

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
