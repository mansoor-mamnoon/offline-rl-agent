"""Generate rollout GIFs from a trained policy in GridWorld or Traffic env."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def collect_rollout(env, policy, max_steps=200, seed=None):
    """Run policy in env, collect frames and metrics."""
    if seed is not None:
        np.random.seed(seed)
    obs = env.reset()
    frames = []
    states, actions, rewards = [obs.copy()], [], []
    done = False
    step = 0

    while not done and step < max_steps:
        action = policy(obs)
        frame = render_frame(env, obs, action, step, rewards)
        frames.append(frame)
        obs, reward, done, info = env.step(action)
        states.append(obs.copy())
        actions.append(action)
        rewards.append(reward)
        step += 1

    return frames, {
        "states": np.array(states[:-1]) if states[:-1] else np.zeros((0, len(obs))),
        "actions": np.array(actions) if actions else np.zeros((0,)),
        "rewards": np.array(rewards) if rewards else np.zeros((0,)),
        "total_return": sum(rewards),
        "n_steps": step,
    }


def render_frame(env, obs, action, step, rewards_so_far):
    """Render a single frame as numpy RGB array."""
    # Try env's own render() first
    if hasattr(env, "render"):
        try:
            frame = env.render()
            if frame is not None and isinstance(frame, np.ndarray):
                return _add_hud_matplotlib(frame, step, rewards_so_far, action)
        except Exception:
            pass
    # Fallback: generate a simple matplotlib frame
    return _matplotlib_frame(obs, action, step, rewards_so_far)


def _add_hud_matplotlib(frame, step, rewards_so_far, action):
    """Add HUD overlay to a frame using matplotlib (no cv2 dependency)."""
    total_r = sum(rewards_so_far) if rewards_so_far else 0.0
    h, w = frame.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, (h + 30) / 100), dpi=100)
    ax.imshow(frame)
    ax.set_title(f"Step: {step}  Return: {total_r:.1f}", fontsize=8, pad=2)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf


def _matplotlib_frame(obs, action, step, rewards_so_far):
    """Generate a 320x240 matplotlib frame as numpy array."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), dpi=50)
    ax_state, ax_action = axes

    obs_show = np.asarray(obs).flatten()
    ax_state.bar(range(min(len(obs_show), 16)), obs_show[:16], color="steelblue")
    ax_state.set_title(f"State (step {step})", fontsize=8)
    ax_state.set_ylim(-1, 1)

    if action is not None:
        act_show = np.asarray(action).flatten()
        ax_action.bar(range(len(act_show)), act_show, color="orange")
        total_r = sum(rewards_so_far) if rewards_so_far else 0.0
        ax_action.set_title(f"Action | R={total_r:.1f}", fontsize=8)
        ax_action.set_ylim(-0.1, 1.1)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf


def save_gif(frames, path: str, fps: int = 10, loop: int = 0):
    """Save list of numpy arrays as animated GIF."""
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        h, w = frames[0].shape[:2]
        frames = [
            f if f.shape[:2] == (h, w) else np.zeros((h, w, 3), dtype=np.uint8)
            for f in frames
        ]
        # Downsample if too many frames
        if len(frames) > 120:
            step = len(frames) // 120
            frames = frames[::step]
        imageio.mimsave(str(path), frames, fps=fps, loop=loop)
    return str(path)


def generate_traffic_comparison_gif(
    env_cls,
    policies: dict,  # name -> select_action callable
    out_path: str,
    n_steps: int = 100,
    seed: int = 42,
):
    """Generate side-by-side comparison GIF for multiple traffic policies."""
    all_rollouts = {}
    for name, policy in policies.items():
        env = env_cls()
        frames, metrics = collect_rollout(env, policy, max_steps=n_steps, seed=seed)
        all_rollouts[name] = (frames, metrics)

    # Make combined frames
    combined = []
    max_len = max(len(f) for f, _ in all_rollouts.values())

    for i in range(max_len):
        row_frames = []
        for name, (frames, metrics) in all_rollouts.items():
            if i < len(frames):
                frame = frames[i].copy()
            else:
                frame = np.zeros_like(frames[0])
            frame_labeled = _add_text_label(frame, name)
            row_frames.append(frame_labeled)
        combined.append(np.concatenate(row_frames, axis=1))

    save_gif(combined, out_path, fps=8)
    return out_path


def _add_text_label(frame, label):
    """Add text label to top of frame using matplotlib."""
    h, w = frame.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, (h + 25) / 100), dpi=100)
    ax.imshow(frame)
    ax.set_title(label, fontsize=9, pad=2)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf
