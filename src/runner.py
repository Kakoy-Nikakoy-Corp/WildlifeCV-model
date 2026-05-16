import os
import subprocess
from dotenv import load_dotenv

load_dotenv()
env = os.environ.copy()


def run_command(cmd):
    process = None

    try:
        process = subprocess.Popen(cmd, env=env)
        return process.wait()

    except KeyboardInterrupt:
        print("\nStopping...")

        if process and process.poll() is None:
            process.terminate()
            process.wait()

        return 0


def serve_ui():
    cmd = [
        "uv",
        "run",
        "mlflow",
        "ui",
        "--backend-store-uri",
        os.environ["MLFLOW_TRACKING_URI"],
        "--default-artifact-root",
        os.environ["ARTIFACT_URI"],
        "--workers",
        "1",
    ]

    run_command(cmd)


def train():
    cmd = [
        "uv",
        "run",
        "python",
        "-X",
        "utf8",
        "src/train.py",
    ]

    run_command(cmd)
