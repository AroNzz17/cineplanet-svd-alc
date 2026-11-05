# modulos/operaciones_matrices.py

def obtener_dimensiones(matriz):
    """Devuelve una tupla (filas, columnas) de una matriz."""
    if not matriz or not matriz[0]:
        return 0, 0
    return len(matriz), len(matriz[0])

def transponer_matriz(matriz):
    """Calcula la transpuesta de una matriz (A^T)."""
    filas, columnas = obtener_dimensiones(matriz)
    # Creamos una nueva matriz invirtiendo filas por columnas
    matriz_T = [[matriz[j][i] for j in range(filas)] for i in range(columnas)]
    return matriz_T

def multiplicar_matrices(A, B):
    """Multiplica dos matrices (A * B) usando el producto punto."""
    filas_A, cols_A = obtener_dimensiones(A)
    filas_B, cols_B = obtener_dimensiones(B)
    
    if cols_A != filas_B:
        raise ValueError(f"No se pueden multiplicar: A tiene {cols_A} columnas y B tiene {filas_B} filas.")
    
    # Crear matriz de resultado llena de ceros
    C = [[0.0 for _ in range(cols_B)] for _ in range(filas_A)]
    
    # Algoritmo clásico de multiplicación de matrices
    for i in range(filas_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
                
    return C

def imprimir_matriz(matriz, nombre="Matriz"):
    """Función auxiliar para ver las matrices bonitas en la consola."""
    print(f"\n{nombre}:")
    for fila in matriz:
        # Formatea cada número con 2 decimales para que se vea ordenado
        print("  [" + ", ".join([f"{num:6.2f}" for num in fila]) + "]")

# Bloque de prueba (solo se ejecuta si corres este archivo directamente)
if __name__ == "__main__":
    # Matriz de prueba (Ejemplo: 3 usuarios, 2 películas)
    matriz_prueba = [
        [5.0, 3.0],
        [4.0, 0.0],
        [1.0, 1.0]
    ]
    
    imprimir_matriz(matriz_prueba, "Matriz Original (A)")
    
    # 1. Probamos transponer
    matriz_transpuesta = transponer_matriz(matriz_prueba)
    imprimir_matriz(matriz_transpuesta, "Matriz Transpuesta (A^T)")
    
    # 2. Probamos multiplicar A^T * A (Paso vital para el SVD)
    matriz_multiplicada = multiplicar_matrices(matriz_transpuesta, matriz_prueba)
    imprimir_matriz(matriz_multiplicada, "Resultado de A^T * A")