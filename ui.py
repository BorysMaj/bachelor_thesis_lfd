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

def run_in_thread(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()

def task_to_env(task_name: str) -> str:
    """Map a UI task name to its robosuite environment name."""
    key = task_name.strip().lower()
    return TASK_TO_ENV.get(key, "Sandbox")


def find_latest_demo(task_name: str):
    """
    Return Path to the most recently modified demo.hdf5 inside data/{task_name}/.
    Returns None if nothing found.
    """
    pattern = str(DEMOS_DIR / task_name / "**" / "demo.hdf5")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return Path(max(matches, key=os.path.getmtime))


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
        f"echo ''; echo 'Collection done - press Enter to close'; read"
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
    Run dataset_states_to_obs then split_train_val in a background thread.
    """
    st.session_state.sim_processing = True
    obs_path = demo_path.parent / "obs.hdf5"
    cwd = str(Path(__file__).parent)

    def _run():
        # 1: convert states to observations
        log(f"Processing: {demo_path.name}")
        log("Step 1 - dataset_states_to_obs")
        cmd1 = [
            "python", "-m", "robomimic.scripts.dataset_states_to_obs",
            "--dataset", str(demo_path),
            "--output_name", "obs.hdf5",
            "--done_mode", "0",
        ]
        proc1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=cwd)
        if proc1.returncode != 0:
            log(f"dataset_states_to_obs failed:\n{proc1.stderr[-500:]}")
            st.session_state.sim_processing = False
            return

        log(f"Saved obs.hdf5 to {obs_path}")
        log("Step 2 - split_train_val")

        # Step 2: split into train/val
        cmd2 = [
            "python", "-m", "robomimic.scripts.split_train_val",
            "--dataset", str(obs_path),
            "--ratio", "0.1",
        ]
        proc2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=cwd)
        if proc2.returncode != 0:
            log(f"split_train_val failed:\n{proc2.stderr[-500:]}")
        else:
            log("Train/val split done. Demos are ready for training.")

        st.session_state.sim_processing = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    
def sim_preview(task_name: str):
    """Show the simulation screenshot for a task if it exists."""
    # Match task name
    key = next((k for k in SIM_PREVIEWS if k.lower() in task_name.lower()), None)
    path = SIM_PREVIEWS.get(key) if key else None

    if path and path.exists():
        st.image(str(path), caption=f"{key} - simulation view", use_container_width=True)
    else:
        # TEMP
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


                    # Controls reminder
                    with st.expander("SpaceMouse controls", expanded=False):
                        st.markdown(
                            "- **Move / tilt** → move end-effector\n"
                            "- **Left button** → hold to close gripper\n"
                            "- **Right button** → reset the demo\n"
                            "- **CTRL + Q** in viewer → quit\n"
                            "- **CTRL + C** in terminal (after collecting all demos) → quit"
                        )

                    if st.button(
                        "▶ Launch Simulation Collection",
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
                            "When you're done, come back here and click **Mark Collection Done**."
                        )
                        if st.button("✓ Mark Collection Done", key="btn_done_sim"):
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
                        rel = latest_demo.relative_to(Path(__file__).parent) \
                              if latest_demo.is_absolute() else latest_demo
                        st.markdown(f"**Found:** `{latest_demo}`")
                        obs_ready = (latest_demo.parent / "obs.hdf5").exists()
                        if obs_ready:
                            st.success("obs.hdf5 already exists for this demo.")
                    else:
                        st.warning("No demo.hdf5 found in this task's data folder yet.")

                    if st.button(
                        "⚙ Process Demos  (states → obs + train/val split)",
                        type="secondary",
                        disabled=st.session_state.sim_processing or latest_demo is None,
                        key="btn_process_sim",
                    ):
                        run_in_thread(process_sim_demos, latest_demo)
                        log("Post-processing started")
                        st.rerun()

                    if st.session_state.sim_processing:
                        st.info("Post-processing in progress - check the Log tab.")

                with col_preview:
                    st.subheader("Environment Preview")
                    sim_preview(task_name)

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
                batch_size = st.number_input("Batch size", value=16, min_value=8)

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
                    log("Training finished" if proc.returncode == 0
                        else f"Training failed (code {proc.returncode})")

                run_in_thread(train)
                st.rerun()

            st.divider()
            st.subheader("Snellius (remote)")
            st.code(
                f"scp {DEMOS_DIR / st.session_state.current_task / 'demos.hdf5'} "
                f"bmajchrzak@snellius.surf.nl:~/thesis/\n"
                f"sbatch ~/thesis/train.job",
                language="bash",
            )

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

            else:
                # Simulation execution
                if st.button("▶ Execute"):

                    task_name = st.session_state.current_task or ""
                st.code(
                    f"python -m robomimic.scripts.run_trained_agent --agent {model_path} --n_rollouts {num_demos_ex} --horizon {horizon} --render",
                    language="bash",
                )

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
