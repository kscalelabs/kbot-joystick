<div align="center">
<h1>Kbot-Joystic</h1>
<p>Train and deploy your own humanoid robot controller in 1k lines of Python</p>
<h3>
  <a href="https://url.kscale.dev/docs">Documentation</a> ·
  <a href="https://github.com/kscalelabs/ksim/tree/master/examples">K-Sim Examples</a>
</h3>

https://github.com/user-attachments/assets/3d44aa23-5ad7-41a3-b467-22165542b8c4

</div>

## Getting Started

You can use this repository as a starting point

1. Clone this repository:

```bash
git clone https://www.github.com/kscalelabs/kbot-joystick
cd kbot-joystick
```

2. Create a new Python environment (we require Python 3.11 or later)
3. Install the package with its dependencies:

```bash
pip install -r requirements.lock
pip install 'jax[cuda12]'  # If using GPU machine, install Jax CUDA libraries
```

4. Train a policy:
Policy converges in about 1hr, but reward keeps going up for about 24hrs
```bash
python -m train
```

5. Convert the checkpoint to a `kinfer` model:

```bash
python -m convert /path/to/ckpt.bin /path/to/model.kinfer
```

6. Visualize the converted model:

```bash
kinfer-sim assets/model.kinfer kbot-headless
```

## Troubleshooting

If you encounter issues, please consult the [ksim documentation](https://docs.kscale.dev/docs/ksim#/) or reach out to us on [Discord](https://url.kscale.dev/docs).

## Tips and Tricks

To see all the available command line arguments, use the command:

```bash
python -m train --help
```

To visualize training your model without using `kscale-mujoco-viewer`, use the command:

```bash
python -m train run_mode=view
```

