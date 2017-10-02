
h <- function(x) 1/(1+exp(-x))
h_deriv <- function(x){
  h(x)*(1-h(x))
}

#Parámetros
theta <- c(0, 0.2, 0, -0.3, -0.5, -1, 2)
x <- 1
## Feed forward
feed_fow <- function(x, theta){
  a_1 <- h(theta[1] + theta[2]*x)
  a_2 <- h(theta[3] + theta[4]*x)
  p <- h(theta[5] + theta[6]*a_1+theta[7]*a_2)
  list(capa_1 = x, capa_2 = c(a_1,a_2), capa_3 = p)
}

val_unidades <- feed_fow(1, theta)
val_unidades
devianza <- function(y, p){
  -(y*log(p) + (1-y)*log(1-p))
}
p <- val_unidades$capa_3[1]
devianza(0, p)

## delta^3 es p-y
delta_3 <- p - 0
#Parciales theta^3_{1,0}, theta^3_{1,1}, theta^3_{1,2}
deriv_3 <- 
  delta_3 * c(1, val_unidades$capa_2)
deriv_3

# Calculamos delta_2 con fórmula de backprop
delta_2 <-  theta[c(6,7)]  * delta_3 * c(h_deriv(theta[1] + theta[2]*x), h_deriv(theta[3] + theta[4]*x))
delta_2

# Parciales theta^2_1_0 y theta^2_1_1, theta^2_1_0 y theta^2_1_1
deriv_2_1 <- delta_2[1] * c(1,val_unidades$capa_1) # parcial theta^2_1_0 y theta^2_1_1
deriv_2_2 <- delta_2[2] * c(1,val_unidades$capa_1) # parcial theta^2_1_0 y theta^2_1_1

gradiente <- c(deriv_2_1, deriv_2_2, deriv_3)
gradiente

## Verificación del gradiente(debe dar muy cercano a la línea anterior)

mat <- diag(7)
for(j in 1:7){
  ep <- 0.000001
  u <- feed_fow(1, theta_1 + ep*mat[,j])$capa_3
  w <- feed_fow(1, theta_1)$capa_3
  dd <- (devianza(0,u) - devianza(0,w))/ep
  print(dd)
}
  