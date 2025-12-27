import re

MAPPINGS = {
    # Buttons
    "Comenzar Ahora": "?go=Limpieza%20de%20Datos",
    "Ver Ejemplo de Análisis": "?go=Estadística%20Descriptiva",
    "Empieza tu Primera Tesis": "?go=Limpieza%20de%20Datos",
    # Cards
    "Estadística Descriptiva": "?go=Estadística%20Descriptiva",
    "Pruebas de Inferencia": "?go=Pruebas%20de%20Inferencia",
    "Modelado Avanzado": "?go=Modelado",
    "Análisis de Supervivencia": "?go=Supervivencia",
    "Tabla 1 Automática": "?go=Tabla%201",
    "Asistente IA": "?go=Asistente%20IA",
    # Modules
    "Limpieza de Datos": "?go=Limpieza%20de%20Datos",
    "Explorador de Datos": "?go=Exploración%20de%20Datos",
    "Ajuste de Tasas": "?go=Ajuste%20de%20Tasas",
    "Tabla 1": "?go=Tabla%201",
    "Pruebas de Inferencia": "?go=Pruebas%20de%20Inferencia",
    "Modelado": "?go=Modelado",
    "Multivariado": "?go=Multivariado",
    "Supervivencia": "?go=Supervivencia",
    "Psicometría": "?go=Psicometría",
    "Asociaciones": "?go=Asociaciones",
    "Concordancia": "?go=Concordancia",
    "Diagnósticos": "?go=Diagnósticos",
    "Gráficos Suite": "?go=Gráficos",
    "Mi Reporte": "?go=Reportes"
}

def get_mapping(text):
    for k, v in MAPPINGS.items():
        if k == text.strip():
            return v
    # Fuzzy match for buttons if needed, but exact is better
    return None

with open('/Users/preciosdeliquidacion/Documents/bio demo/raw_home.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Separate Style and HTML
style_pattern = re.compile(r'<style>(.*?)</style>', re.DOTALL)
style_match = style_pattern.search(content)
css_content = style_match.group(1) if style_match else ""
html_content = style_pattern.sub('', content).strip()

# --- CSS PROCESSING ---
css_lines = [".biometric-home a { text-decoration: none; color: inherit; }"]
# We will do a line-by-line simple parse assuming standard formatting
for line in css_content.split('\n'):
    sline = line.strip()
    if not sline or sline.startswith('/*') or sline.startswith('--') or sline.startswith('@') or sline.startswith('}'):
        css_lines.append(line)
        continue
    
    if '{' in line:
        parts = line.split('{')
        selectors = parts[0].split(',')
        new_selectors = []
        for sel in selectors:
            sel = sel.strip()
            if not sel: continue
            if sel.startswith('.biometric-home') or 'from' in sel or 'to' in sel or '%' in sel:
                new_selectors.append(sel)
            else:
                new_selectors.append(f".biometric-home {sel}")
        css_lines.append(", ".join(new_selectors) + " {" + "{".join(parts[1:]))
    else:
        css_lines.append(line)

final_css = "\n".join(css_lines)

# --- HTML PROCESSING VIA STATE MACHINE ---
lines = html_content.split('\n')
output_lines = []
buffer = []
state = "NORMAL"

# Indentation check for closing tags
# Card: 6 spaces
# Module: 8 spaces => <div class="module-card"> is at 8 usually?
# Let's inspect the input file indentation dynamically or assume it fits the prompt copy-paste.
# The prompt copy-paste has:
#    <div class="grid grid-3">
#      
#      <div class="card">
# So card is at 6 spaces.
#         <div class="module-card">
# Module is at 8 spaces.

for line in lines:
    stripped = line.strip()
    
    if state == "NORMAL":
        if '<div class="card">' in line:
            state = "CARD"
            buffer = [line]
        elif '<div class="module-card">' in line:
            state = "MODULE"
            buffer = [line]
        elif '<button class="btn' in line:
            # Process button inline
            # text is likely inside or on next line?
            # Buttons in this file are multiline.
            # <button ...>
            #   <span>...
            # </button>
            # Let's verify.
            #     <button class="btn btn-primary">
            #       <span class="btn-icon">🚀</span>
            #       <span>Comenzar Ahora</span>
            #     </button>
            # It's a block!
            if '</button>' in line: # Single line button?
                 attr = re.search(r'class="btn(.*?)"', line).group(1)
                 txt = re.sub(r'<.*?>', '', line).strip()
                 target = MAPPINGS.get(txt)
                 # Fuzzy for button
                 if "Comenzar Ahora" in line: target = MAPPINGS["Comenzar Ahora"]
                 if "Ver Ejemplo" in line: target_line = MAPPINGS["Ver Ejemplo de Análisis"]
                 if target:
                     line = line.replace('<button', '<a').replace('</button>', '</a>')
                     line = line.replace('class="btn', f'target="_self" href="{target}" class="btn')
                 output_lines.append(line)
            else:
                 state = "BUTTON"
                 buffer = [line]
        else:
             output_lines.append(line)

    elif state == "CARD":
        buffer.append(line)
        # Check for closing div
        # Leading spaces = 6
        leading_spaces = len(line) - len(line.lstrip())
        if stripped == "</div>" and leading_spaces == 6:
            # Process Buffer
            block = "\n".join(buffer)
            # Extact title
            title_match = re.search(r'<h3 class="card-title">(.*?)</h3>', block)
            title = title_match.group(1) if title_match else ""
            target = MAPPINGS.get(title)
            
            if target:
                # Replace wrapper
                block = block.replace('<div class="card">', f'<a class="card" href="{target}" target="_self">', 1)
                # Last div closure
                block = block[:block.rfind('</div>')] + '</a>'
                # Inner link
                block = re.sub(r'<a href="#" class="card-link">(.*?)</a>', r'<div class="card-link">\1</div>', block, flags=re.DOTALL)
            output_lines.append(block)
            buffer = []
            state = "NORMAL"
            
    elif state == "MODULE":
        buffer.append(line)
        leading_spaces = len(line) - len(line.lstrip())
        # Module indented at 8 spaces usually?
        # Let's check raw file indentation in buffer[0]
        start_spaces = len(buffer[0]) - len(buffer[0].lstrip())
        if stripped == "</div>" and leading_spaces == start_spaces:
            block = "\n".join(buffer)
            name_match = re.search(r'<p class="module-name">(.*?)</p>', block)
            name = name_match.group(1) if name_match else ""
            target = MAPPINGS.get(name)
            
            if target:
                block = block.replace('<div class="module-card">', f'<a class="module-card" href="{target}" target="_self">', 1)
                block = block[:block.rfind('</div>')] + '</a>'
            
            output_lines.append(block)
            buffer = []
            state = "NORMAL"

    elif state == "BUTTON":
        buffer.append(line)
        if '</button>' in line:
            block = "\n".join(buffer)
            # Find destination
            target = None
            if "Comenzar Ahora" in block: target = MAPPINGS["Comenzar Ahora"]
            elif "Ver Ejemplo" in block: target = MAPPINGS["Ver Ejemplo de Análisis"]
            elif "Empieza tu Primera Tesis" in block: target = MAPPINGS["Empieza tu Primera Tesis"]
            
            if target:
                block = block.replace('<button', '<a').replace('</button>', '</a>')
                # Inject href
                # naive replace of class attribute to append href?
                # better: replace the opening tag
                block = re.sub(r'<a(.*?)>', f'<a\\1 href="{target}" target="_self">', block, 1)
            
            output_lines.append(block)
            buffer = []
            state = "NORMAL"


print("<style>")
print(final_css)
print("</style>")
print("\n".join(output_lines))

final_output = "<style>
" + final_css + "
</style>
" + "
".join(output_lines)
with open('home_styled_final.html', 'w', encoding='utf-8') as f: f.write(final_output)
