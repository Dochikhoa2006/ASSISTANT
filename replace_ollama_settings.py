import sys

def replace_in_file():
    with open("assistant_rag/settings.py", "r") as f:
        lines = f.readlines()
    
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("class OllamaSettings:"):
            start_idx = i - 1
            break
            
    end_idx = -1
    for i in range(start_idx + 1, len(lines)):
        if line.startswith("class ") or line.startswith("@dataclass"):
            # Wait, better logic to find end of class: it's until the next @dataclass
            if lines[i].startswith("@dataclass"):
                end_idx = i
                break
                
    if start_idx == -1 or end_idx == -1:
        print("Could not find class bounds")
        sys.exit(1)
        
    with open("generate_settings.py", "r") as f:
        gen_script = f.read()
    
    # execute generate script and capture output to split it
    import io
    from contextlib import redirect_stdout
    out = io.StringIO()
    with redirect_stdout(out):
        exec(gen_script)
    
    generated = out.getvalue()
    parts = generated.split("================================================================================")
    class_def = parts[0].strip() + "\n\n\n"
    from_env_def = parts[2].strip() + "\n"
    
    new_lines = lines[:start_idx] + [class_def] + lines[end_idx:]
    
    # Now find from_env
    env_start = -1
    for i, line in enumerate(new_lines):
        if "ollama=OllamaSettings(" in line:
            env_start = i
            break
            
    env_end = -1
    for i in range(env_start + 1, len(new_lines)):
        if new_lines[i].strip() == ")," and "ollama" not in new_lines[i]:
            # Actually, looking for the exact indentation of `),` for `ollama=OllamaSettings(`
            if new_lines[i].startswith("            ),"):
                env_end = i + 1
                break
                
    if env_start == -1 or env_end == -1:
        print("Could not find from_env bounds")
        sys.exit(1)
        
    final_lines = new_lines[:env_start] + [from_env_def] + new_lines[env_end:]
    
    with open("assistant_rag/settings.py", "w") as f:
        f.write("".join(final_lines))
        
replace_in_file()
