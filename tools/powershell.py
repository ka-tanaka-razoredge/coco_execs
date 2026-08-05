import subprocess

def run_powershell(command: str):

    result = subprocess.run(
        ["powershell.exe", "-Command", command],
        capture_output=True,
        text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }
