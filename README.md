# Simulación del Modelo SIR

Este repositorio contiene una simulación del modelo SIR (Susceptible, Infectado, Recuperado) en diferentes configuraciones espaciales: circular y rectangular, con y sin clústeres.

## Estructura del Proyecto

- `Circulo.py`: Implementación del modelo SIR en una distribución circular uniforme.
- `Circulo_cluster.py`: Implementación del modelo SIR en una distribución circular con clústeres.
- `Rectangulo.py`: Implementación del modelo SIR en una distribución rectangular uniforme.
- `Rectangulo_Cluster.py`: Implementación del modelo SIR en una distribución rectangular con clústeres.

## Requisitos
- Python 3
- Bibliotecas: numpy, pandas, matplotlib,scipy
- R

```sh
pip install numpy pandas matplotlib scipy
```
Este paquete se puede instalar desde CRAN
```sh
install.packages("shinySIR")
```
## Ciudad Rectangular
![Ciudad Rectangular](Animaciones_gif/sir_simulationreactangle.gif)


## Ciudad Rectangular con cluster

![Ciudad Rectangular con cluster](Animaciones_gif/sir_simulation_rectangulo_cluster.gif)

## Ciudad Circular

![Ciudad Circular](Animaciones_gif/sir_simulation_circulo.gif)

## Ciudad Circular con cluster

![Ciudad Circular con cluster](Animaciones_gif/sir_simulation_circulo_cluster.gif)
