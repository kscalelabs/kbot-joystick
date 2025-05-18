<div align="center">
<h1>K-Bot Joystick Example</h1>
<p>Train and deploy your own humanoid robot controller in 700 lines of Python by checking out <a href="https://github.com/kscalelabs/ksim-gym">K-Sim Gym</a></p>

<img src="https://github.com/kscalelabs/kbot-joystick/raw/master/assets/forward.gif" alt="K-Bot Joystick Demo" width="250">

<img src="https://github.com/kscalelabs/kbot-joystick/raw/master/assets/turn.gif" alt="K-Bot Joystick Demo" width="250">

</div>

## Overview
This repository contains an example of a joystick controller for the K-Bot humanoid robot. It currently takes significantly longer than the K-Sim Gym example to train (~3000 steps vs ~150 steps).

If you think you can bring that time down though, have a crack at it!

The main differences between this example and the K-Sim Gym benchmark are:
1. Joystick commands as a one-hot encoded vector with 7 dimensions:
    - 0: Stand still
    - 1: Move forward
    - 2: Move backward
    - 3: Move left
    - 4: Move right
    - 5: Turn left
    - 6: Turn right
2. A lot more randomization in observations and actions
3. Phase-based feet height tracking reward to encourage a more natural gait

You can also try out the pre-trained and converted model in `assets/joystick.kinfer` by:
1. Cloning the repository
2. Installing `kinfer-sim` with `pip install kinfer-sim`
3. Running the following command:
```bash
kinfer-sim assets/joystick.kinfer kbot --use-keyboard
```
4. Focus on the terminal window that's running `kinfer-sim` and use `W`, `A`, `S`, `D`, `Q`, `E` to move the robot around.

https://github.com/user-attachments/assets/422b7d58-1cc0-4ac6-b8ea-4d63f3c21fce

## Getting Started

### On your own GPU

1. Read through the [current leaderboard](https://url.kscale.dev/leaderboard) submissions and through the [ksim examples](https://github.com/kscalelabs/ksim/tree/master/examples)
2. Make sure you have installed `git-lfs`:

```bash
sudo apt install git-lfs  # Ubuntu
brew install git-lfs  # MacOS
```

4. Clone this repository:

```bash
git clone git@github.com:kscalelabs/kbot-joystick.git
cd kbot-joystick
```

5. Create a new Python environment (we require Python 3.11 or later)
6. Install the package with its dependencies:

```bash
pip install -e .
pip install 'jax[cuda12]'  # If using GPU machine, install Jax CUDA libraries
```

7. Train a policy:

```bash
python -m train
```

8. Convert the checkpoint to a `kinfer` model:

```bash
python -m convert /path/to/ckpt.bin /path/to/model.kinfer
```

9. Visualize the converted model:

```bash
kinfer-sim assets/joystick.kinfer kbot --save-video assets/video.mp4
```

10. Commit the K-Infer model and the recorded video to this repository
11. Push your code and model to your repository, and make sure the repository is public
12. Write a message with a link to your repository on our [Discord](https://url.kscale.dev/discord) in the "【🧠】submissions" channel
13. Wait for one of us to run it on the real robot - this should take about a day, but if we are dragging our feet, please message us on Discord
14. Voila! Your name will now appear on our [leaderboard](https://url.kscale.dev/leaderboard)

## Troubleshooting

If you encounter issues, please consult the [ksim documentation](https://docs.kscale.dev/docs/ksim#/) or reach out to us on [Discord](https://url.kscale.dev/docs).

## Tips and Tricks

To see all the available command line arguments, use the command:

```bash
python -m train --help
```

To visualize running your model without using `kos-sim`, use the command:

```bash
python -m train run_mode=view
```

This repository contains a pre-trained checkpoint of a model which has been learned to be robust to pushes, which is useful for both jump-starting model training and understanding the codebase. To initialize training from this checkpoint, use the command:

```bash
python -m train load_from_ckpt_path=assets/ckpt.bin
```

You can visualize the pre-trained model by combining these two commands:

```bash
python -m train load_from_ckpt_path=assets/ckpt.bin run_mode=view
```

