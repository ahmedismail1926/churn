with open('data_splitting.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode characters with ASCII
content = content.replace('✓', '[OK]')
content = content.replace('⚠', '[WARNING]')
content = content.replace('█', '#')

with open('data_splitting.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed Unicode characters in data_splitting.py')
