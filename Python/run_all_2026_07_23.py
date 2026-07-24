import subprocess
import sys

scripts = [
    'train_full_test_disjoint_2026_07_23.py',
    'analyze_full_2026_07_23_disjoint.py',
    'train_full_2026_07_23.py',
    'analyze_full_2026_07_23.py'
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, script], stdout=None, stderr=None)
    if result.returncode != 0:
        print(f"{script} failed with exit code {result.returncode}, continuing...\n")
    else:
        print(f"{script} completed\n")

print("All scripts completed!")