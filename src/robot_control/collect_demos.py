import time
import threading
import os
import numpy as np
from demo_recorder import KinestheticDemoRecorder

ROBOT_IP  = "172.16.0.2"
RECORD_HZ = 20
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# Reach task table bounds (robot base frame, metres)
TABLE_X_MIN = 0.25
TABLE_X_MAX = 0.60
TABLE_Y_MIN = -0.25 # wall side (with saftey margin)
TABLE_Y_MAX = 0.35
TABLE_Z = 0.037 # hover height measured at table corners
REACH_THRESHOLD = 0.05 # 5 cm success


# Helpers
def sample_reach_target(rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = rng.uniform(TABLE_X_MIN, TABLE_X_MAX)
    y = rng.uniform(TABLE_Y_MIN, TABLE_Y_MAX)
    return np.array([x, y, TABLE_Z], dtype=np.float32)


def ascii_target(target):
    """
    Print a top-down ASCII map of the table with the target marked as X.

    Orientation (matches what the human sees standing at the bottom edge):
     - Top = wall (robot Y_MIN, negative Y)
     - Bottom = human (robot Y_MAX, positive Y)
     - Left = far from robot (robot X_MAX)
     - Right = near robot (robot X_MIN)
    Robot is shown on the right edge.
    """
    W, H = 25, 11
    tx = (target[0] - TABLE_X_MIN) / (TABLE_X_MAX - TABLE_X_MIN)
    ty = (target[1] - TABLE_Y_MIN) / (TABLE_Y_MAX - TABLE_Y_MIN)
    col = int((1 - tx) * (W - 1)) # X_MIN (robot side) -> right
    row = int(ty * (H - 1)) # Y_MIN (wall) -> top, Y_MAX (human) -> bottom

    print("\n  Table - top-down view")
    print("  WALL ↑")
    print("  ┌" + "─" * W + "┐")
    for r in range(H):
        line = ""
        for c in range(W):
            if r == row and c == col:
                line += "X"
            else:
                line += "·"
        # robot marker on the right edge at mid-height
        robot_marker = " [R]" if r == H // 2 else "    "
        print(f"  │{line}│{robot_marker}")
    print("  └" + "─" * W + "┘")
    print("  ↓ you")
    print(f"\nTarget: x={target[0]:.3f}, y={target[1]:.3f}, z={target[2]:.3f} (metres)\n")


def recording_loop(recorder):
    """Standard recording loop (non-reach tasks)."""
    while recorder.is_recording:
        recorder.record_step()
        time.sleep(recorder.dt)


def reach_recording_loop(recorder, success_flag):
    """
    Recording loop for reach task.
    Monitors distance to target - auto-stops 1 second after success.
    success_flag is a threading.Event set when target is reached.
    """
    success_printed = False
    success_time    = None

    while recorder.is_recording:
        dist = recorder.record_step_reach()
        time.sleep(recorder.dt)

        if dist < REACH_THRESHOLD and not success_printed:
            print(f"\n Target Reached, dist={dist*100:.1f} cm - stopping in 1 s.")
            success_printed = True
            success_time = time.time()
            success_flag.set()

        if success_time is not None and (time.time() - success_time) >= 1.0:
            recorder.is_recording = False
            break


# Main

def main():
    print("Demo Collector")

    # Task name
    task_name = input("\nEnter task name: ").strip()
    if not task_name:
        task_name = "unnamed_task"
    task_name = task_name.replace(" ", "_").lower()

    save_path = os.path.join(DATA_DIR, task_name, "demos.hdf5")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Data will be saved to: {save_path}")

    # Connect
    recorder = KinestheticDemoRecorder(
        robot_ip=ROBOT_IP,
        record_hz=RECORD_HZ
    )
    recorder.connect()

    rng = np.random.default_rng()

    while True:
        print(f"\nTask: {task_name} | Demos recorded: {len(recorder.demos)}")
        print("  [r]  Record new demo")
        print("  [rr] Record reach demo (random target + auto-stop on success)")
        print("  [h]  Go home")
        print("  [m]  Measure a position")
        print("  [s]  Save and quit")
        print("  [q]  Quit without saving")

        choice = input("\nChoice: ").strip().lower()

        # Standard record
        if choice == "r":
            print("\nMoving to home position.")
            recorder.move_to_home()

            input("Press Enter to start recording.")
            recorder.enable_teaching_mode()

            recorder.start_recording()

            thread = threading.Thread(target=recording_loop, args=(recorder,))
            thread.start()

            input("Press Enter when done.")

            recorder.stop_recording()
            thread.join()
            recorder.disable_teaching_mode()

            T = recorder.demos[-1]["actions"].shape[0] if recorder.demos else 0
            keep = input("Keep this demo? (y/n): ").strip().lower()
            if keep != "y":
                recorder.demos.pop()
                print("Demo discarded.")
            else:
                print(f"Demo {len(recorder.demos) - 1} kept  ({T} steps).")

            recorder.move_to_home()

        # Reach record
        elif choice == "rr":

            # Sample and display target
            target = sample_reach_target(rng)
            ascii_target(target)
            input("Place a marker at that position, then press Enter to start.")

            recorder.enable_teaching_mode()
            recorder.start_recording_reach(target)

            success_flag = threading.Event()
            thread = threading.Thread(
                target=reach_recording_loop,
                args=(recorder, success_flag)
            )
            thread.start()

            print("Recording, move the EEF to the target.")
            print(" (Press Enter at any time to stop early.)\n")

            # Wait for either manual stop or auto-stop on success
            stop_input = threading.Event()
            def wait_for_enter():
                input()
                stop_input.set()

            enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
            enter_thread.start()

            while thread.is_alive():
                if stop_input.is_set():
                    recorder.is_recording = False
                    break
                time.sleep(0.05)

            thread.join()
            recorder.stop_recording()
            recorder.disable_teaching_mode()

            demo = recorder.demos[-1]
            T = demo["actions"].shape[0]
            reached = success_flag.is_set()
            status = "SUCCESS" if reached else "no success"
            print(f"\n  Demo: {T} steps - {status}")

            keep = input("Keep this demo? (y/n): ").strip().lower()
            if keep != "y":
                recorder.demos.pop()
                print("Demo discarded.")
            else:
                print(f"Demo {len(recorder.demos) - 1} kept.")

            recorder.move_to_home()

        # Go home
        elif choice == "h":
            print("Moving to home.")
            recorder.move_to_home()

        # Measure
        elif choice == "m":
            input("Press Enter to enable free move.")
            recorder.enable_teaching_mode()
            input("Press Enter when EEF is at desired position.")
            recorder.disable_teaching_mode()

            recorder.get_possition()
            recorder.move_to_home()

        # Save
        elif choice == "s":
            if not recorder.demos:
                print("No demos to save.")
                continue

            print(f"\nSaving {len(recorder.demos)} demos to {save_path}.")
            recorder.save(save_path, task_name=task_name)
            print("Done!")

            print("Moving to home.")
            recorder.move_to_home()
            break

        # Quit
        elif choice == "q":
            if recorder.demos:
                confirm = input(f"You have {len(recorder.demos)} unsaved demos. Save first? (y/n): ").strip().lower()
                if confirm == "y":
                    recorder.save(save_path, task_name=task_name)
                    print("Saved.")
                    
            print("Moving to home.")
            recorder.move_to_home()

            print("Goodbye!")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()