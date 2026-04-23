import time
import threading
import os
import sys
from demo_recorder import KinestheticDemoRecorder

ROBOT_IP  = "172.16.0.2"
RECORD_HZ = 20
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def recording_loop(recorder):
    """Background thread - records at fixed Hz while is_recording=True."""
    while recorder.is_recording:
        recorder.record_step()
        time.sleep(recorder.dt)


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

    # Main loop
    while True:
        print(f"Task: {task_name} | Demos recorded: {len(recorder.demos)}")
        print("[r] Record new demo")
        print("[h] Go home")
        print("[s] Save and quit")
        print("[q] Quit without saving")

        choice = input("\nChoice: ").strip().lower()

        # Record
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

            demo = recorder.demos[-1]
            print(demo["obs"]["robot0_gripper_qpos"][-1])
            T = demo["actions"].shape[0]

            keep = input("Keep this demo? (y/n): ").strip().lower()
            if keep != "y":
                recorder.demos.pop()
                print("Demo discarded.")
            else:
                print(f"Demo {len(recorder.demos) - 1} kept.")

            print("Moving to home.")
            recorder.move_to_home()

        # Go home
        elif choice == "h":
            print("Moving to home.")
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