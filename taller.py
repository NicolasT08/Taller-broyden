import numpy as np

def broyden(f, x0):
    tol=1e-14
    max_iter=200
    n = len(x0)
    x = x0
    B = np.eye(n)
    fx = f(x)
    
    for i in range(max_iter):
        dx = np.linalg.solve(B, -fx)
        
        newx = x + dx
        newfx = f(newx)
        
        if np.linalg.norm(newfx) < tol:
            print(f"iteraciones =  {i+1}")
            return newx
        
        delta_x = newx - x
        delta_f = newfx - fx
        
        B += np.outer((delta_f - B @ delta_x), delta_x) / np.dot(delta_x, delta_x)
        
        x = newx
        fx = newfx
    
    print(f'diverge')
    return x

def newton_raphson(f, J, x0):
    tol=1e-14
    max_iter=200
    
    x = x0
    for i in range(max_iter):
        fx = f(x)
        Jx = J(x)
        
        dx = np.linalg.solve(Jx, -fx)
        x_new = x + dx
        
        if np.linalg.norm(f(x_new)) < tol:
            print(f"iteraciones =  {i+1}")
            return x_new
        
        x = x_new 
    
    print(f'diverge')
    return x



def sistema_ecuaciones(V):
    V1, V2, V3 = V
    return np.array([
        V1**2 - V2 + V3 - 3,
        V1 - V2**3 + np.cos(V3) - 1,
        np.sin(V1) + V2 - V3**2  
    ])

x0 = np.array([1.0, 1.0, 1.0])

solucion = broyden(sistema_ecuaciones, x0)
print(f'La solucion aproximada por broyden es V1 = {solucion[0]}, V2 = {solucion[1]}, V3 = {solucion[2]}')

def jacobiano(V):
    V1, V2, V3 = V
    return np.array([
        [2*V1, -1, 1],
        [1, -3*V2**2, -np.sin(V3)],
        [np.cos(V1), 1, -2*V3]
    ])

solucion = newton_raphson(sistema_ecuaciones, jacobiano, x0)

print(f'Solución aproximada por newton raphson V1 = {solucion[0]}, V2 = {solucion[1]}, V3 = {solucion[2]}')

