from environments.environment import Environment
import tasks
import time


env = Environment(
        "/Users/maxfest/vscode/thesis/ravens/environments/assets",
        disp=True,
        shared_memory=False,
        hz=480,
        record_cfg={
            "save_video": False,
            "save_video_path": "${data_dir}/${task}-cap/videos/",
            "add_text": True,
            "add_task_text": True,
            "fps": 20,
            "video_height": 640,
            "video_width": 720
         }
    )

task = tasks.names["put-block-in-bowl"]()
env.set_task(task)
env.reset()

for _ in range(1000):
    time.sleep(1)
