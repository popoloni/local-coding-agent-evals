import subprocess


def run_user_command(command):
    return subprocess.check_output(command, shell=True, text=True)
