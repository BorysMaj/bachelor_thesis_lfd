"""
Robot LfD UI - Streamlit interface for demo recording, training, and policy execution.
Run with: streamlit run ui.py
"""

import streamlit as st
import subprocess
import threading
import queue
import time
import os
import glob
import json
from pathlib import Path

# Log buffer - background threads write here, main thread drains it
_log_queue: queue.Queue = queue.Queue()

# Config
ROBOT_IP      = "172.16.0.2"
DEMOS_DIR     = Path("data")
MODELS_DIR    = Path("models")
CONFIG_DIR    = Path("config")
ASSETS_DIR    = Path(__file__).parent / "assets"
RECORDER_PATH = Path(__file__).parent / "src/robot_control/demo_recorder.py"
EXECUTE_PATH  = Path(__file__).parent / "src/learning/execute_policy.py"

# Maps UI task names to robosuite env names
TASK_TO_ENV = {
    "reach":      "ReachTask",
    "push":       "PushTask",
    "lift":       "Lift",
    "stack":      "Stack",
    "wave":       "Playground",
    "playground": "Playground",
}

DEMOS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

SIM_PREVIEWS = {
    "Reach":  ASSETS_DIR / "sim_reach.png",
    "Push":   ASSETS_DIR / "sim_push.png",
    "Lift":   ASSETS_DIR / "sim_lift.png",
    "Stack":  ASSETS_DIR / "sim_stack.png",
    "Sandbox":  ASSETS_DIR / "sim_sandbox.png",
}

# Session state

def init_state():
    defaults = {
        "robot_connected":     False,
        "recording":           False,
        "training":            False,
        "executing":           False,
        "current_task":        None,
        "mode":                "Real Robot",   # Real Robot or Simulation
        "log":                 [],
        "recorder":            None,
        "sim_collecting":      False,   # True while collection terminal is open
        "sim_last_demo_path":  None,    # path to most recent demo.hdf5
        "sim_processing":      False,   # True while post-processing runs
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# Logging

def log(msg: str):
    """Log a message. Call from background threads."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    try:
        st.session_state.log.append(entry)
        if len(st.session_state.log) > 100:
            st.session_state.log = st.session_state.log[-100:]
    except Exception:
        _log_queue.put(entry)

def drain_log_queue():
    """Drain queued log messages into session state. Call at the top of main()."""
    while not _log_queue.empty():
        try:
            entry = _log_queue.get_nowait()
            st.session_state.log.append(entry)
        except queue.Empty:
            break
    if len(st.session_state.log) > 100:
        st.session_state.log = st.session_state.log[-100:]

# Helpers
def get_tasks():
    if not DEMOS_DIR.exists():
        return []
    return sorted([d.name for d in DEMOS_DIR.iterdir() if d.is_dir()])

def get_demo_count(task: str) -> int:
    import h5py
    total = 0

    # Real robot demos - demos.hdf5 directly in task folder
    real_path = DEMOS_DIR / task / "demos.hdf5"
    if real_path.exists():
        try:
            with h5py.File(real_path, "r") as f:
                total += len(f["data"].keys())
        except Exception:
            pass

    # Sim demos - demo.hdf5 inside timestamped subdirectories
    sim_pattern = str(DEMOS_DIR / task / "**" / "demo.hdf5")
    for sim_demo in glob.glob(sim_pattern, recursive=True):
        try:
            with h5py.File(sim_demo, "r") as f:
                total += len(f["data"].keys())
        except Exception:
            pass

    return total

def get_models(task: str):
    pattern = str(MODELS_DIR / task / "**" / "*.pth")
    return sorted(glob.glob(pattern, recursive=True))


def find_latest_checkpoint(task: str) -> Path | None:
    """Return path to last.pth for the most recently modified training run, or None."""
    pattern = str(MODELS_DIR / task / "**" / "last.pth")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return Path(max(matches, key=os.path.getmtime))


def get_epoch_from_checkpoint(ckpt_path: Path) -> int | None:
    """Read the last completed epoch number from a robomimic checkpoint."""
    try:
        import torch as _torch
        ckpt = _torch.load(str(ckpt_path), map_location="cpu")
        return int(ckpt["variable_state"]["epoch"])
    except Exception:
        return None

def run_in_thread(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()

def task_to_env(task_name: str) -> str:
    """Map a UI task name to its robosuite environment name."""
    key = task_name.strip().lower()
    return TASK_TO_ENV.get(key, "Sandbox")


BASE_CONFIG_PATH = Path(__file__).parent / "src/learning/train_bc_rnn.json"


def generate_train_config(task_name: str, dataset_path: Path,
                          n_epochs: int, batch_size: int,
                          resume_from_epoch: int | None = None) -> Path:
    """
    Read the base train_bc_rnn.json, patch the fields that change per run,
    and write the result to config/<task_name>/bc_rnn.json.

    If resume_from_epoch is provided, num_epochs is set to
    resume_from_epoch + n_epochs so --resume continues for exactly
    n_epochs more steps.

    Returns the path to the generated config.
    """
    import json as _json

    with open(BASE_CONFIG_PATH, "r") as f:
        cfg = _json.load(f)

    output_dir = str((MODELS_DIR / task_name).resolve())

    cfg["train"]["data"]       = [{"path": str(dataset_path.resolve())}]
    cfg["train"]["output_dir"] = output_dir
    cfg["train"]["batch_size"] = int(batch_size)

    if resume_from_epoch is not None:
        cfg["train"]["num_epochs"] = int(resume_from_epoch) + int(n_epochs)
    else:
        cfg["train"]["num_epochs"] = int(n_epochs)

    out_path = CONFIG_DIR / task_name / "bc_rnn.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        _json.dump(cfg, f, indent=4)

    return out_path


def find_latest_demo(task_name: str):
    """
    Return Path to the most recently modified demo.hdf5 inside data/{task_name}/.
    Excludes the merged folder.
    Returns None if nothing found.
    """
    pattern = str(DEMOS_DIR / task_name / "**" / "demo.hdf5")
    matches = [
        p for p in glob.glob(pattern, recursive=True)
        if "merged" not in Path(p).parts
    ]
    if not matches:
        return None
    return Path(max(matches, key=os.path.getmtime))


def find_all_hdf5(task_name: str) -> list[Path]:
    """
    Return all .hdf5 files for a task (obs.hdf5, merged files, etc).
    Sorted by modification time, newest first.
    """
    pattern = str(DEMOS_DIR / task_name / "**" / "*.hdf5")
    matches = glob.glob(pattern, recursive=True)
    return sorted([Path(p) for p in matches], key=os.path.getmtime, reverse=True)


def get_merged_path(task_name: str) -> Path:
    """Return the canonical path for the merged demo file."""
    return DEMOS_DIR / task_name / "merged" / "merged.hdf5"


def launch_sim_collection(task_name: str, env_name: str):
    """
    Open a new terminal window running robosuite collection script.
    Uses collect_human_demonstrations.py whith custom and already avaiable envs.
    Returns the Popen object for the terminal.
    """
    output_dir = str((DEMOS_DIR / task_name).resolve())
    cmd_inner = (
        f"cd {Path(__file__).parent.resolve()} && "
        f"sudo /home/borys/miniconda3/envs/franka/bin/python -m robosuite.scripts.collect_human_demonstrations "
        f"--environment {env_name} "
        f"--robots Panda "
        f"--device spacemouse "
        f"--directory {output_dir}; "
        f"echo ''; echo 'Fixing file ownership...'; "
        f"sudo chown -R $USER:$USER {output_dir}; "
        f"echo 'Collection done - press Enter to close'; read"
    )
    # Try gnome-terminal first, fall back to xterm
    try:
        proc = subprocess.Popen(
            ["gnome-terminal", "--", "bash", "-c", cmd_inner],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        proc = subprocess.Popen(
            ["xterm", "-e", f"bash -c '{cmd_inner}'"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return proc


def process_sim_demos(demo_path: Path):
    """
    Run dataset_states_to_obs then split_train_val in a new gnome-terminal window.
    Both commands run sequentially in one terminal so the user can watch progress.
    """
    obs_path = demo_path.parent / "obs.hdf5"

    cmd1 = [
        "python", "-m", "robomimic.scripts.dataset_states_to_obs",
        "--dataset", str(demo_path),
        "--output_name", "obs.hdf5",
        "--done_mode", "0",
    ]
    cmd2 = [
        "python", "-m", "robomimic.scripts.split_train_val",
        "--dataset", str(obs_path),
        "--ratio", "0.1",
    ]

    cmd1_str = " ".join(cmd1)
    cmd2_str = " ".join(cmd2)
    bash_cmd = (
        f"{cmd1_str} && {cmd2_str} && "
        f"echo '' && echo '--- Processing complete! ---' || "
        f"echo '' && echo '--- Processing FAILED. Check errors above. ---'; "
        f"echo 'Press Enter to close...'; read"
    )

    subprocess.Popen(
        ["gnome-terminal", "--", "bash", "-c", bash_cmd],
        cwd=str(Path(__file__).parent)
    )
    log("Processing started in a new terminal window.")
    log("When done, click Refresh to check if obs.hdf5 is ready.")
    st.session_state.sim_processing = False
    
def sim_preview(task_name: str):
    """Show the simulation screenshot for a task or Sandbox env."""
    # Match task name
    key = next((k for k in SIM_PREVIEWS if k.lower() in task_name.lower()), None)
    path = SIM_PREVIEWS.get(key) if key else None
    sandbox_path = SIM_PREVIEWS.get("Sandbox")

    if path and path.exists():
        st.image(str(path), caption=f"{key} - simulation view", width='content')
    elif sandbox_path.exists():
        sandbox_path = SIM_PREVIEWS.get("Sandbox")
        st.image(str(sandbox_path), caption="Sandbox - simulation view", width='content')
    else:
        st.markdown(
            """
            <div style="
                border: 2px dashed #555;
                border-radius: 8px;
                padding: 40px;
                text-align: center;
                color: #888;
                background: #1a1a1a;
            ">
                Simulation preview<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Main UI 

def main():
    init_state()
    drain_log_queue()

    st.set_page_config(
        page_title="Robot LfD",
        page_icon="🤖",
        layout="wide",
    )

    st.title("Robot Learning from Demonstration")

    # Sidebar 
    with st.sidebar:

        # Mode toggle - top of sidebar, always visible
        st.header("Mode")
        mode = st.radio(
            "Environment",
            ["Real Robot", "Simulation"],
            index=0 if st.session_state.mode == "Real Robot" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            log(f"Switched to {mode} mode")
            st.rerun()

        if st.session_state.mode == "Real Robot":
            st.markdown("**Real Robot** - kinesthetic teaching on Franka")
        else:
            st.markdown("**Simulation** - robosuite SpaceMouse collection")

        st.divider()

        # Robot connection (only relevant in real robot mode)
        if st.session_state.mode == "Real Robot":
            st.header("Connection")
            robot_color = "🟢" if st.session_state.robot_connected else "🔴"
            st.markdown(
                f"{robot_color} {'Connected' if st.session_state.robot_connected else 'Disconnected'}"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Connect", disabled=st.session_state.robot_connected):
                    try:
                        import panda_py
                        from panda_py import libfranka
                        st.session_state.panda   = panda_py.Panda(ROBOT_IP)
                        st.session_state.gripper = libfranka.Gripper(ROBOT_IP)
                        st.session_state.robot_connected = True
                        log(f"Connected to Franka at {ROBOT_IP}")
                    except Exception as e:
                        log(f"Connection failed: {e}")
            with col2:
                if st.button("Disconnect", disabled=not st.session_state.robot_connected):
                    st.session_state.robot_connected = False
                    st.session_state.panda   = None
                    st.session_state.gripper = None
                    log("Disconnected")

            st.divider()

        # Mode indicator
        if st.session_state.recording:
            st.markdown("🔴 **Recording**")
        elif st.session_state.training:
            st.markdown("🔵 **Training**")
        elif st.session_state.executing:
            st.markdown("🟡 **Executing**")
        else:
            st.markdown("⚪ **Idle**")

        st.divider()

        # Task management
        st.header("Tasks")
        tasks = get_tasks()

        new_task = st.text_input("New task name")
        if st.button("Create task") and new_task.strip():
            task_dir = DEMOS_DIR / new_task.strip()
            task_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.current_task = new_task.strip()
            log(f"Created task: {new_task.strip()}")
            st.rerun()

        if tasks:
            selected = st.selectbox(
                "Select task",
                tasks,
                index=tasks.index(st.session_state.current_task)
                      if st.session_state.current_task in tasks else 0,
            )
            if selected != st.session_state.current_task:
                st.session_state.current_task = selected
                log(f"Switched to task: {selected}")

            if st.session_state.current_task:
                n = get_demo_count(st.session_state.current_task)
                st.metric("Demos recorded", n)
        else:
            st.info("No tasks yet. Create one above.")

    # Main tabs 
    tab_record, tab_train, tab_execute, tab_log = st.tabs(
        ["⏺ Record", "⚙ Train", "▶ Execute", "📋 Log"]
    )

    # RECORD tab 
    with tab_record:
        st.header("Record Demonstrations")

        if not st.session_state.current_task:
            st.warning("Select or create a task in the sidebar first.")
        else:
            st.markdown(f"**Task:** `{st.session_state.current_task}`  |  "
                        f"**Demos recorded:** {get_demo_count(st.session_state.current_task)}")

            if st.session_state.mode == "Real Robot":
                # Real robot recording
                st.info(
                    "1. Click **Start Recording**\n"
                    "2. Perform the demonstration\n"
                    "3. Click **Stop Recording**"
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "⏺ Start Recording",
                        disabled=st.session_state.recording or not st.session_state.robot_connected,
                        type="primary",
                    ):
                        try:
                            from src.robot_control.demo_recorder import KinestheticDemoRecorder
                            rec = KinestheticDemoRecorder(robot_ip=ROBOT_IP)
                            rec.panda = st.session_state.panda
                            rec.gripper = st.session_state.gripper
                            rec.enable_teaching_mode()
                            rec.start_recording()
                            st.session_state.recorder = rec
                            st.session_state.recording = True
                            log("Recording started")

                            def poll():
                                while st.session_state.recording:
                                    st.session_state.recorder.record_step()
                                    time.sleep(0.05)
                            run_in_thread(poll)
                        except Exception as e:
                            log(f"Recording error: {e}")

                with col2:
                    if st.button(
                        "■ Stop Recording",
                        disabled=not st.session_state.recording,
                        type="secondary",
                    ):
                        st.session_state.recording = False
                        time.sleep(0.1)
                        try:
                            rec  = st.session_state.recorder
                            demo = rec.stop_recording()
                            rec.disable_teaching_mode()
                            T = demo["actions"].shape[0]
                            save_path = DEMOS_DIR / st.session_state.current_task / "demos.hdf5"
                            rec.save(str(save_path), task_name=st.session_state.current_task)
                            log(f"Saved demo - {T} steps ({T/20:.1f}s)")
                            st.session_state.recorder = None
                        except Exception as e:
                            log(f"Stop recording error: {e}")
                        st.rerun()

                if st.session_state.recording:
                    st.error("🔴 Recording in progress...")

            else:
                # Simulation recording
                task_name = st.session_state.current_task
                env_name  = task_to_env(task_name)

                col_info, col_preview = st.columns([1, 1])

                with col_info:
                    st.subheader("Collect in Simulation")

                    # Step by step instruction
                    st.success("**Step by step instruction for collection**\n"
                        " 1. Read the how to use SpaceMouse\n"
                        " 2. Press 'Start Collecting Demonstrations' to open simulation\n"
                        " 3. After collection press 'Collection Done' button\n"
                        " 4. Process the demonstration file in the 'Post-process Demos' section\n"
                        " 5. After Proccessing is done press the refresh button to check if everything went good\n"
                        " 6. If you collected demos to an existing task, merge the files in section bellow"
                    )

                    # Controls instruction
                    with st.expander("SpaceMouse controls", expanded=False):
                        st.markdown(
                            "- **Move / tilt** → move end-effector\n"
                            "- **Left button** → hold to close gripper\n"
                            "- **Right button** → reset the demo\n"
                            "- **Both buttons** → save the demo\n"
                            "- **CTRL + Q** in viewer → quit\n"
                            "- **CTRL + C** in terminal (after collecting all demos) → quit"
                        )

                    if st.button(
                        "▶ Start Collecting Demonstrations",
                        type="primary",
                        disabled=st.session_state.sim_collecting,
                        key="btn_launch_sim",
                    ):
                        (DEMOS_DIR / task_name).mkdir(parents=True, exist_ok=True)
                        try:
                            launch_sim_collection(task_name, env_name)
                            st.session_state.sim_collecting = True
                            log(f"Launched sim collection - env={env_name}")
                            log(f"Output will appear in data/{task_name}/TEMP/demo.hdf5")
                        except Exception as e:
                            log(f"Failed to launch terminal: {e}")

                    if st.session_state.sim_collecting:
                        st.info(
                            "🟢 Collection terminal and simulation window is open.\n\n"
                            "When you're done, come back here and click **Collection Done**."
                        )
                        if st.button("✓ Collection Done", key="btn_done_sim"):
                            st.session_state.sim_collecting = False
                            # Auto-detect the latest demo.hdf5
                            latest = find_latest_demo(task_name)
                            if latest:
                                st.session_state.sim_last_demo_path = str(latest)
                                log(f"Detected demo file: {latest}")
                            else:
                                log("No demo.hdf5 found yet - check the data folder.")
                            st.rerun()

                    st.divider()

                    # Post-processing
                    st.subheader("Post-process Demos")

                    # Auto-detect or show current
                    latest_demo = find_latest_demo(task_name)
                    if latest_demo:
                        st.markdown(f"**Latest batch:** `{latest_demo}`")
                        obs_ready = (latest_demo.parent / "obs.hdf5").exists()
                        if obs_ready:
                            st.success("obs.hdf5 already exists for this batch.")
                    else:
                        st.warning("No demo.hdf5 found in this task's data folder yet.")

                    col_proc, col_refresh = st.columns([3, 1])
                    with col_proc:
                        if st.button(
                            "⚙ Process Demos  (states → obs + train/val split)",
                            type="secondary",
                            disabled=st.session_state.sim_processing or latest_demo is None,
                            key="btn_process_sim",
                        ):
                            process_sim_demos(latest_demo)
                            log("Post-processing started")
                            st.rerun()
                    with col_refresh:
                        if st.button("🔄 Refresh", key="btn_refresh_obs", help="Check if processing finished and update the merge file list"):
                            st.rerun()

                    if st.session_state.sim_processing:
                        st.info("Post-processing in progress - check the Log tab.")

                    st.divider()

                    # Merge demos
                    st.subheader("Merge Demo Batches")

                    all_obs = [p for p in find_all_hdf5(task_name) if p.name == "obs.hdf5"]
                    merged_path = get_merged_path(task_name)

                    # Only count obs files that are not yet in the merged file
                    if merged_path.exists():
                        merged_mtime = merged_path.stat().st_mtime
                        new_obs = [p for p in all_obs if p.stat().st_mtime > merged_mtime]
                        st.info(f"Existing merged file found — {len(new_obs)} new batch(es) since last merge.")
                    else:
                        new_obs = all_obs

                    if len(all_obs) == 0:
                        st.warning("No obs.hdf5 files found. Process demos first.")
                    elif len(new_obs) == 0:
                        st.success("Merged file is already up to date.")
                    elif len(new_obs) == 1 and not merged_path.exists():
                        st.info("Only one batch found. Collect more demos to merge.")
                    else:
                        st.markdown(f"Found **{len(new_obs)}** new batch(es) to merge.")
                        if st.button("🔀 Merge All Batches", key="btn_merge_sim", type="secondary"):
                            merged_path.parent.mkdir(parents=True, exist_ok=True)
                            # Only use new obs files + existing merged (not all obs files)
                            sources = list(new_obs)
                            if merged_path.exists():
                                sources.append(merged_path)

                            merge_script = str(Path(__file__).parent / "merge_demos.py")
                            # Pass all obs files + existing merged to a temp merged, then replace.
                            tmp_merged = str(merged_path.parent / "merged_tmp.hdf5")
                            src_args = " ".join(f'"{str(s)}"' for s in sources)
                            bash_cmd = (
                                f"python {merge_script} "
                                f"--inputs {src_args} "
                                f"--out \"{str(merged_path)}\" && "
                                f"echo '' && echo '--- Merge complete! ---' || "
                                f"echo '' && echo '--- Merge FAILED. ---'; "
                                f"echo 'Press Enter to close...'; read"
                            )
                            subprocess.Popen(
                                ["gnome-terminal", "--", "bash", "-c", bash_cmd],
                                cwd=str(Path(__file__).parent)
                            )
                            log(f"Merging {len(sources)} files into {merged_path}")

                with col_preview:
                    st.subheader("Environment Preview")
                    sim_preview(task_name)

    # TRAIN tab
    with tab_train:
        st.header("Train Policy")

        st.success("**How to train a policy**\n"
                        " 1. If you already have recorded demonstrations choose which file you want to train on.\n"
                        " 2. Choose the hyperparameters.\n"
                        " - Epochs - how many times the model trains through your demos. More = longer training but better learning. Start with 500.\n"
                        " - Batch size - how many steps the model learns from at once. Keep at 16 unless you get memory issues errors.\n"
                        " 3. When ready press 'Train Localy'. Training will take at least 10 minutes.\n"
                        " 4. After training you can see the performance of policy in 'Execute' tab\n"
                    )

        if not st.session_state.current_task:
            st.warning("Select a task first.")
        else:
            n_demos = get_demo_count(st.session_state.current_task)
            st.markdown(f"**Task:** `{st.session_state.current_task}` - {n_demos} demos")

            if n_demos < 15:
                st.warning(f"Only {n_demos} demos. Recommend at least 20 before training.")

            st.subheader("Local training")

            # Dataset picker - any .hdf5 in the task folder
            available_hdf5 = find_all_hdf5(st.session_state.current_task)
            # Filter to files that look like processed obs files (not raw demo.hdf5)
            trainable = [p for p in available_hdf5 if p.name != "demo.hdf5"]
            if trainable:
                hdf5_labels = [str(p.relative_to(DEMOS_DIR)) for p in trainable]
                selected_label = st.selectbox("Dataset to train on", hdf5_labels, index=0)
                dataset_path = DEMOS_DIR / selected_label
            else:
                st.warning("No processed .hdf5 files found. Run Post-process first.")
                dataset_path = None

            # Check for existing checkpoint
            last_ckpt = find_latest_checkpoint(st.session_state.current_task)
            last_epoch = get_epoch_from_checkpoint(last_ckpt) if last_ckpt else None
            is_resume = last_epoch is not None

            if is_resume:
                st.info(
                    f"**Continued training** - an existing model was found "
                    f"(trained up to epoch {last_epoch}). "
                    f"Training will resume from epoch {last_epoch + 1} and run for the "
                    f"number of additional epochs you set below."
                )

            col1, col2 = st.columns(2)
            with col1:
                epoch_label = "Additional epochs" if is_resume else "Epochs"
                n_epochs = st.number_input(epoch_label, value=100 if is_resume else 500, min_value=10, step=50)
            with col2:
                batch_size = st.number_input("Batch size", value=16, min_value=8, step=4)

            if is_resume:
                st.caption(f"Total epochs after training: **{last_epoch + n_epochs}**")

            btn_label = "⚙ Continue Training" if is_resume else "⚙ Train locally"
            if st.button(
                btn_label,
                disabled=st.session_state.training or n_demos == 0 or dataset_path is None,
                type="primary",
            ):
                config_path = generate_train_config(
                    task_name=st.session_state.current_task,
                    dataset_path=dataset_path,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    resume_from_epoch=last_epoch,
                )
                log(f"Config written to {config_path}")

                def train():
                    cmd = [
                        "python", "-m", "robomimic.scripts.train",
                        "--config", str(config_path),
                    ]
                    if is_resume:
                        cmd.append("--resume")
                    cmd_str = " ".join(cmd)
                    bash_cmd = (
                        f"{cmd_str} && echo '' && echo '--- Training complete! ---' || "
                        f"echo '' && echo '--- Training FAILED. ---'; "
                        f"echo 'Press Enter to close...'; read"
                    )

                    subprocess.Popen(
                        ["gnome-terminal", "--", "bash", "-c", bash_cmd],
                        cwd=str(Path(__file__).parent)
                    )
                    extra = f"(continuing from epoch {last_epoch})" if is_resume else ""
                    log(f"Training started in a new terminal - {n_epochs} epochs {extra}")
                    st.session_state.training = False

                train()
                st.rerun()

            st.divider()

            models = get_models(st.session_state.current_task)
            if models:
                st.subheader("Saved checkpoints")
                for m in models[-5:]:
                    st.text(Path(m).name)

    # EXECUTE tab 
    with tab_execute:
        st.header("Execute Policy")

        models = get_models(st.session_state.current_task) if st.session_state.current_task else []

        if not models:
            st.warning("No trained models found for this task.")
        else:
            model_path = st.selectbox(
                "Select checkpoint",
                models,
                format_func=lambda p: Path(p).name,
                index=len(models) - 1,
            )

            col1, col2 = st.columns(2)
            with col1:
                horizon = st.number_input("Horizon (steps)", value=400, min_value=50, step=50)
            with col2:
                num_demos_ex = st.number_input("Number of demos executed", value=10, min_value=1, step=1)

            if st.session_state.mode == "Real Robot":
                st.warning("⚠ Make sure the workspace is clear before running.")

                col_run, col_stop = st.columns(2)
                with col_run:
                    if st.button(
                        "▶ Run on Robot",
                        disabled=st.session_state.executing or not st.session_state.robot_connected,
                        type="primary",
                    ):
                        def execute():
                            cmd = [
                                "python", "-u", str(EXECUTE_PATH),
                                "--policy", model_path,
                                "--horizon", str(horizon),
                            ]
                            cmd_str = " ".join(cmd)
                            bash_cmd = (
                                f"{cmd_str} && echo '' && echo '--- Execution complete! ---' || "
                                f"echo '' && echo '--- Execution FAILED. ---'; "
                                f"echo 'Press Enter to close...'; read"
                            )

                            subprocess.Popen(
                                ["gnome-terminal", "--", "bash", "-c", bash_cmd],
                                cwd=str(Path(__file__).parent)
                            )
                            log(f"Policy execution started in a new terminal.")
                            st.session_state.executing = False

                        execute()
                        st.rerun()

                with col_stop:
                    if st.button("■ Stop", disabled=not st.session_state.executing, type="secondary"):
                        st.session_state.executing = False
                        log("Execution stopped by user")

                if st.session_state.executing:
                    st.warning("🟡 Policy running...")

            else:
                # Simulation execution
                if st.button("▶ Execute in Simulation"):
                    cmd = [
                        "python", "-m", "robomimic.scripts.run_trained_agent",
                        "--agent", model_path,
                        "--n_rollouts", str(num_demos_ex),
                        "--horizon", str(horizon),
                        "--render",
                    ]
                    cmd_str = " ".join(cmd)
                    bash_cmd = (
                        f"{cmd_str} && echo '' && echo '--- Execution complete! ---' || "
                        f"echo '' && echo '--- Execution FAILED. ---'; "
                        f"echo 'Press Enter to close...'; read"
                    )

                    subprocess.Popen(
                        ["gnome-terminal", "--", "bash", "-c", bash_cmd],
                        cwd=str(Path(__file__).parent)
                    )
                    log("Sim execution started in a new terminal.")

            # Feedback
            if not st.session_state.executing:
                st.divider()
                st.subheader("Rate last execution")
                col_good, col_bad = st.columns(2)
                with col_good:
                    if st.button("👍 Worked", type="primary"):
                        log("Execution rated: GOOD")
                with col_bad:
                    if st.button("👎 Failed"):
                        log("Execution rated: BAD - consider adding more demos")

    # LOG tab
    with tab_log:
        st.header("Activity Log")
        if st.button("Clear log"):
            st.session_state.log = []
        log_text = "\n".join(reversed(st.session_state.log)) if st.session_state.log else "No activity yet."
        st.text_area("Log", value=log_text, height=400, label_visibility="collapsed")


if __name__ == "__main__":
    main()