import re

with open(r'C:\Users\Dell\Downloads\pmf\config\requirements.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    pkg = re.split(r'==|>=|<=|~=|<|>', line)[0]
    new_lines.append(pkg)

with open(r'C:\Users\Dell\Downloads\pmf\config\requirements.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines) + '\n')
