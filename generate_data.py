
import pandas as pd
import numpy as np
import random

def generar_datos_demo(n=100):
    np.random.seed(42)
    random.seed(42)
    
    ids = range(1, n + 1)
    grupos = np.random.choice(['Control', 'Tratamiento'], n)
    edades = np.random.randint(18, 90, n)
    sexos = np.random.choice(['M', 'F'], n)
    
    # Simular datos con cierta correlación para que las pruebas estadísticas tengan sentido
    glucosa = []
    presion = []
    desenlaces = []
    
    for i in range(n):
        # Grupo Tratamiento tiende a tener mejor glucosa y presión
        base_glucosa = 100 if grupos[i] == 'Control' else 90
        base_presion = 130 if grupos[i] == 'Control' else 120
        
        g = int(np.random.normal(base_glucosa, 15))
        p = int(np.random.normal(base_presion, 10))
        
        glucosa.append(g)
        presion.append(p)
        
        # Desenlace depende un poco de la glucosa y grupo
        prob_curado = 0.3
        if grupos[i] == 'Tratamiento':
            prob_curado += 0.3
        if g < 100:
            prob_curado += 0.1
            
        prob_curado = min(prob_curado, 0.9)
        desenlace = 'Curado' if random.random() < prob_curado else 'No Curado'
        desenlaces.append(desenlace)
        
    df = pd.DataFrame({
        'ID': ids,
        'Grupo': grupos,
        'Edad': edades,
        'Sexo': sexos,
        'Glucosa': glucosa,
        'Presión_Arterial': presion,
        'Desenlace_Clinico': desenlaces
    })
    
    df.to_excel('datos_pacientes_demo.xlsx', index=False)
    print("Archivo 'datos_pacientes_demo.xlsx' generado exitosamente.")

if __name__ == "__main__":
    generar_datos_demo()
