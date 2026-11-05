# modulos/algoritmo_svd.py
import math
from modulos.operaciones_matrices import multiplicar_matrices, transponer_matriz, obtener_dimensiones

# ==========================================
# 1. FUNCIONES AUXILIARES PARA VECTORES
# ==========================================

def multiplicar_matriz_vector(matriz, vector):
    """Multiplica una matriz por un vector unidimensional."""
    filas = len(matriz)
    columnas = len(matriz[0])
    resultado = [0.0 for _ in range(filas)]
    
    for i in range(filas):
        for j in range(columnas):
            resultado[i] += matriz[i][j] * vector[j]
    return resultado

def norma_vector(vector):
    """Calcula la norma (longitud) de un vector."""
    suma_cuadrados = sum(x**2 for x in vector)
    return math.sqrt(suma_cuadrados)

def normalizar_vector(vector):
    """Convierte un vector en un vector unitario (norma 1)."""
    norma = norma_vector(vector)
    if norma == 0:
        return vector
    return [x / norma for x in vector]

def producto_punto_vectores(v1, v2):
    """Calcula el producto punto entre dos vectores."""
    return sum(x * y for x, y in zip(v1, v2))

def restar_vectores(v1, v2):
    """Resta el vector v2 al vector v1."""
    return [x - y for x, y in zip(v1, v2)]

def multiplicar_escalar_vector(escalar, vector):
    """Multiplica un vector por un número (escalar)."""
    return [escalar * x for x in vector]

# ==========================================
# 2. EL MOTOR MATEMÁTICO: MÉTODO DE LAS POTENCIAS Y GRAM-SCHMIDT
# ==========================================

def metodo_potencias(A, iteraciones=100, tolerancia=1e-7):
    """
    Encuentra el autovalor dominante y su autovector correspondiente
    mediante iteraciones multiplicativas.
    """
    n = len(A)
    # Inicializamos un vector de unos
    v = [1.0 for _ in range(n)]
    v = normalizar_vector(v)
    
    autovalor = 0.0
    for _ in range(iteraciones):
        v_nuevo = multiplicar_matriz_vector(A, v)
        autovalor_nuevo = norma_vector(v_nuevo)
        
        # Evitar división por cero
        if autovalor_nuevo == 0:
            break
            
        v_nuevo = normalizar_vector(v_nuevo)
        
        # Condición de parada si ya convergemos a la respuesta
        if abs(autovalor_nuevo - autovalor) < tolerancia:
            break
            
        autovalor = autovalor_nuevo
        v = v_nuevo
        
    return autovalor, v

def gram_schmidt(vectores):
    """
    Aplica el proceso de ortogonalización de Gram-Schmidt
    para asegurar que un conjunto de vectores formen una base ortonormal.
    """
    vectores_ortogonales = []
    for v in vectores:
        v_orto = v.copy()
        # Restamos la proyección de v sobre todos los vectores ortogonales previos
        for u in vectores_ortogonales:
            proyeccion = producto_punto_vectores(v, u)
            v_proyectado = multiplicar_escalar_vector(proyeccion, u)
            v_orto = restar_vectores(v_orto, v_proyectado)
        
        # Normalizamos el vector resultante y lo guardamos
        vectores_ortogonales.append(normalizar_vector(v_orto))
    return vectores_ortogonales

# ==========================================
# 3. ENSAMBLAJE DE LA SVD
# ==========================================

def calcular_svd_truncado(matriz_R, k_dimensiones=2):
    """
    Calcula una aproximación de SVD (Truncated SVD) utilizando los k valores 
    singulares más grandes. Retorna las matrices U_k, Sigma_k y Vt_k.
    """
    print("\nIniciando factorización SVD manual...")
    matriz_Rt = transponer_matriz(matriz_R)
    
    # Matriz simétrica Rt * R (Para hallar V)
    Rt_R = multiplicar_matrices(matriz_Rt, matriz_R)
    
    autovalores = []
    autovectores_V = []
    
    # Usamos deflación para hallar los k componentes principales
    matriz_residual = [fila.copy() for fila in Rt_R]
    
    for _ in range(k_dimensiones):
        # 1. Hallar el autovalor y autovector dominante
        val, vec = metodo_potencias(matriz_residual)
        autovalores.append(val)
        autovectores_V.append(vec)
        
        # 2. Deflación: restar el componente hallado de la matriz residual 
        # para que en el siguiente ciclo encuentre el SEGUNDO más grande, y así sucesivamente.
        for i in range(len(matriz_residual)):
            for j in range(len(matriz_residual[0])):
                matriz_residual[i][j] -= val * vec[i] * vec[j]
                
    # 3. Ortogonalizar los vectores V (Gram-Schmidt)
    autovectores_V = gram_schmidt(autovectores_V)
    
    # 4. Calcular Valores Singulares (Sigma)
    # Son la raíz cuadrada de los autovalores
    valores_singulares = [math.sqrt(abs(val)) for val in autovalores]
    
    # 5. Calcular matriz U
    # U_i = (R * V_i) / sigma_i
    autovectores_U = []
    for i in range(k_dimensiones):
        R_vi = multiplicar_matriz_vector(matriz_R, autovectores_V[i])
        sigma_i = valores_singulares[i]
        
        if sigma_i > 1e-7:
            u_i = multiplicar_escalar_vector(1.0 / sigma_i, R_vi)
        else:
            u_i = [0.0 for _ in range(len(matriz_R))]
        autovectores_U.append(u_i)
        
    # Las matrices U y V están formadas por vectores columna, así que las transponemos
    U_k = transponer_matriz(autovectores_U)
    Vt_k = autovectores_V # Ya son vectores fila (transpuestos)
    
    return U_k, valores_singulares, Vt_k

# Bloque de prueba local
if __name__ == "__main__":
    # Matriz de valoraciones simulada (4 usuarios, 3 productos)
    R_prueba = [
        [5.0, 3.0, 0.0],
        [4.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 5.0]
    ]
    
    U, Sigma, Vt = calcular_svd_truncado(R_prueba, k_dimensiones=2)
    
    print("\n--- Resultados SVD ---")
    print("Valores Singulares (Sigma):", [round(s, 4) for s in Sigma])
    print("Las matrices U y Vt han sido calculadas con éxito.")