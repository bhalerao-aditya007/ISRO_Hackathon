import glob
for f in glob.glob('agents/*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace('from base_agent import', 'from core.base_agent import')
    content = content.replace('from protocol import', 'from core.protocol import')
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
print("Imports fixed safely.")
