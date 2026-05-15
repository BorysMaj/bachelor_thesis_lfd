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
ROBOT_IP = "172.16.0.2"
DEMOS_DIR = Path("data").expanduser()
MODELS_DIR = Path("models").expanduser()
CONFIG_DIR = Path("configs").expanduser()
RECORDER_PATH = Path(__file__).parent / "src/robot_control/demo_recorder.py"
EXECUTE_PATH  = Path(__file__).parent / "src/learning/execute_policy.py"

DEMOS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Session state defaults
def init_state():
    defaults = {
        "robot_connected": False,
        "recording": False,
        "training": False,
        "executing": False,
        "current_task": None,
        "log": [],
        "recorder": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log(msg: str):
    """Log a message."""
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    try:
        # Works when called from the main Streamlit thread
        st.session_state.log.append(entry)
        if len(st.session_state.log) > 100:
            st.session_state.log = st.session_state.log[-100:]
    except Exception:
        # Called from a background thread — queue it for the main thread
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
    """List task names = subdirectories of DEMOS_DIR."""
    if not DEMOS_DIR.exists():
        return []
    return sorted([d.name for d in DEMOS_DIR.iterdir() if d.is_dir()])

def get_demo_count(task: str) -> int:
    path = DEMOS_DIR / task / "demos.hdf5"
    if not path.exists():
        return 0
    try:
        import h5py
        with h5py.File(path, "r") as f:
            return len(f["data"].keys())
    except Exception:
        return 0

def get_models(task: str):
    pattern = str(MODELS_DIR / task / "**" / "*.pth")
    return sorted(glob.glob(pattern, recursive=True))

def run_in_thread(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()

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
        st.header("Status")

        # Robot connection
        robot_color = "🟢" if st.session_state.robot_connected else "🔴"
        st.markdown(f"{robot_color} Robot: {'Connected' if st.session_state.robot_connected else 'Disconnected'}")

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
            st.markdown("🔴 **Mode: Recording**")
        elif st.session_state.training:
            st.markdown("🔵 **Mode: Training**")
        elif st.session_state.executing:
            st.markdown("🟡 **Mode: Executing**")
        else:
            st.markdown("⚪ **Mode: Idle**")

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
            st.markdown(f"**Task:** `{st.session_state.current_task}`")
            st.markdown(f"**Demos so far:** {get_demo_count(st.session_state.current_task)}")

            st.info(
                "1. Pre-position the arm at the start of the motion\n"
                "2. Click **Start Recording**\n"
                "3. Perform the demonstration\n"
                "4. Click **Stop Recording**"
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
                        rec.panda   = st.session_state.panda
                        rec.gripper = st.session_state.gripper
                        rec.enable_teaching_mode()
                        rec.start_recording()
                        st.session_state.recorder  = rec
                        st.session_state.recording = True
                        log("Recording started")

                        # Start background polling at 20 Hz
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

                        # Save to HDF5
                        save_path = DEMOS_DIR / st.session_state.current_task / "demos.hdf5"
                        rec.save(str(save_path), task_name=st.session_state.current_task)
                        log(f"Saved demo - {T} steps ({T/20:.1f}s) → {save_path}")
                        st.session_state.recorder = None
                    except Exception as e:
                        log(f"Stop recording error: {e}")
                    st.rerun()

            if st.session_state.recording:
                st.error("🔴 Recording in progress...")

    # TRAIN tab
    with tab_train:
        st.header("Train Policy")

        if not st.session_state.current_task:
            st.warning("Select a task first.")
        else:
            n_demos = get_demo_count(st.session_state.current_task)
            st.markdown(f"**Task:** `{st.session_state.current_task}` - {n_demos} demos")

            if n_demos < 5:
                st.warning(f"Only {n_demos} demos. Recommend at least 20 before training.")

            st.subheader("Local training")
            col1, col2 = st.columns(2)
            with col1:
                n_epochs = st.number_input("Epochs", value=500, min_value=100, step=100)
            with col2:
                batch_size = st.number_input("Batch size", value=100, min_value=16)

            if st.button(
                "⚙ Train locally",
                disabled=st.session_state.training or n_demos == 0,
                type="primary",
            ):
                dataset_path = DEMOS_DIR / st.session_state.current_task / "demos.hdf5"
                output_dir   = MODELS_DIR / st.session_state.current_task
                config_path = CONFIG_DIR / st.session_state.current_task / "bc_rnn.json"

                def train():
                    st.session_state.training = True
                    log(f"Training started - {n_epochs} epochs")
                    cmd = [
                        "python", "-m", "robomimic.scripts.train",
                        "--config", str(config_path),
                        "--dataset", str(dataset_path),
                        "--output_dir", str(output_dir),
                    ]
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True)
                    for line in proc.stdout:
                        line = line.strip()
                        if any(k in line for k in ["Loss", "Epoch", "loss", "error"]):
                            log(line)
                    proc.wait()
                    st.session_state.training = False
                    log("Training finished" if proc.returncode == 0 else f"Training failed (code {proc.returncode})")

                run_in_thread(train)
                st.rerun()

            st.divider()
            st.subheader("Snellius (remote)")
            st.code(
                f"scp {DEMOS_DIR / st.session_state.current_task / 'demos.hdf5'} "
                f"bmajchrzak@snellius.surf.nl:~/thesis/\nsbatch ~/thesis/train.job",
                language="bash"
            )

            # Model list
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
                horizon = st.number_input("Horizon (steps)", value=200, min_value=50, step=5)
            with col2:
                action_scale = st.slider("Action scale", 0.1, 2.0, 1.0, 0.1)

            st.warning("⚠ Make sure the workspace is clear before running.")

            col_run, col_stop = st.columns(2)
            with col_run:
                if st.button(
                    "▶ Run Policy",
                    disabled=st.session_state.executing or not st.session_state.robot_connected,
                    type="primary",
                ):
                    def execute():
                        st.session_state.executing = True
                        log(f"Running policy: {Path(model_path).name}")
                        cmd = [
                            "python", "-u", str(EXECUTE_PATH),
                            "--policy", model_path,
                            "--horizon", str(horizon),
                        ]
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT, text=True)
                        for line in proc.stdout:
                            log(line.strip())
                        proc.wait()
                        st.session_state.executing = False
                        log("Execution finished")

                    run_in_thread(execute)
                    st.rerun()

            with col_stop:
                if st.button("■ Stop", disabled=not st.session_state.executing, type="secondary"):
                    st.session_state.executing = False
                    log("Execution stopped by user")

            if st.session_state.executing:
                st.warning("🟡 Policy running...")

            # Feedback
            if not st.session_state.executing and len(get_models(st.session_state.current_task or "")) > 0:
                st.divider()
                st.subheader("Rate last execution")
                col_good, col_bad = st.columns(2)
                with col_good:
                    if st.button("👍 Good", type="primary"):
                        log("Execution rated: GOOD")
                with col_bad:
                    if st.button("👎 Bad"):
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