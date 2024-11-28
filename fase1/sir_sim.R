library("devtools")
library(shinySIR)
sir_model <- function(time, state, parameters) {  
  with(as.list(c(state, parameters)), {# R obtendrá los nombres de variables a
    # partir de inputs de estados y parametros
    N <- S+I+R 
    lambda <- beta * I/N
    dS <- -lambda * S              
    dI <- lambda * S - gamma * I   
    dR <- gamma * I             
    return(list(c(dS, dI, dR))) 
  })
}

run_shiny(model = "Modelo SIR", 
          neweqns = sir_model,
          ics = c(S = 4500, I = 1, R = 0),
          parm0 = c(beta = 1, gamma = 1),
          parm_names = c("Razón de Transmisión", "Razón de recuperación"),
          parm_min = c(beta = 0, gamma = 0),
          parm_max = c(beta = 1.2, gamma = 1),
          tmax=200,
          legend_title = "Subconjunto",
          xlabel="Tiempo (dias)",
          ylabel="Numero de casos")



