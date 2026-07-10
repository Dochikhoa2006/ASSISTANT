with open("debug_pipeline.py", "r") as f:
    lines = f.readlines()

with open("debug_pipeline.py", "w") as f:
    for i, line in enumerate(lines):
        if 816 <= i <= 975:
            continue
        f.write(line)
