import subprocess
import os
import csv
PROGRAMS = ["desafio.py", "ear.py", "naive.py"]

# Define se a categoria é real ou spoofing
REAL_VIDEO = {
    'real': True,
    'foto': False,
    'foto_impressa': False,
    'video': False,
    'video_spooging_completo': False
}

IMAGES_PATH = '../images'
RESULTS_FILE = 'results.csv'

PERSONS = os.listdir(IMAGES_PATH)

results = []

for person in PERSONS:
    print(f"Iniciando os videos de {person} ...")
    person_path = os.path.join(IMAGES_PATH, person)

    if not os.path.isdir(person_path):
        continue

    foto_path = os.path.join(person_path, 'foto.jpeg')

    if not os.path.exists(foto_path):
        print(f'Imagem não encontrada para {person}')
        continue

    # Percorre cada categoria (real, foto, video, etc.)
    for category in os.listdir(person_path):
        print(f"Processando a categoria {category}")
        category_path = os.path.join(person_path, category)

        if not os.path.isdir(category_path):
            continue

        # Percorre todos os arquivos dentro da categoria
        for file in os.listdir(category_path):
            if not file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(category_path, file)

            for program in PROGRAMS:
                process = subprocess.run(
                    ["python", program, foto_path, video_path],
                    capture_output=True,
                    text=True
                )

                output = process.stdout.strip()

                results.append({
                    'person': person,
                    'program': program.replace('.py', ''),
                    'category': category,
                    'video': file,
                    'is_real': REAL_VIDEO.get(category, False),
                    'output': output
                })

# Salva CSV
with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['person', 'program', 'category', 'video', 'is_real', 'output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print('Processamento concluído.')