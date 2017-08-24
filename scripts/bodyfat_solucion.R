
library(tidyr)
library(dplyr)

dat_grasa <- read_csv(file = 'datos/bodyfat.csv')
head(dat_grasa)
nrow(dat_grasa)
set.seed(127)
dat_grasa$unif <- runif(nrow(dat_grasa), 0, 1)
dat_grasa <- arrange(dat_grasa, unif)
dat_grasa$id <- 1:nrow(dat_grasa)
dat_e <- dat_grasa[1:150,]
dat_p <- dat_grasa[151:252,]

set.seed(127)
dat_grasa$unif <- runif(nrow(dat_grasa), 0, 1)
dat_grasa <- arrange(dat_grasa, unif)
dat_grasa$id <- 1:nrow(dat_grasa)
dat_e <- dat_grasa[1:150,]
dat_p <- dat_grasa[151:252,]

normalizacion <- dat_e %>% 
  gather(variable, valor, edad:muñeca) %>%
  group_by(variable) %>%
  summarise(media = mean(valor), de = sd(valor))

dat_e_norm <- dat_e %>% 
  gather(variable, valor, edad:muñeca) %>%
  left_join(normalizacion) %>%
  mutate(valor_norm = (valor - media)/de) %>%
  select(id, grasacorp, variable, valor_norm) %>%
  spread(variable, valor_norm) 

#######################
#2. Completa estas funciones

rss_calc <- function(x, y){
  # x es un data.frame o matrix con entradas
  # y es la respuesta
  rss_fun <- function(beta){
    # esta funcion debe devolver rss
    y_hat <- as.matrix(cbind(1,x)) %*% beta
    e <- y - y_hat
    rss <- 0.5*sum(e^2)
    rss
  }
  rss_fun
}


grad_calc <- function(x, y){
  # devuelve una función que calcula el gradiente para 
  # parámetros beta   
  # x es un data.frame o matrix con entradas
  # y es la respuesta
  grad_fun <- function(beta){
      f_beta <- as.matrix(cbind(1, x)) %*% beta
      e <- y - f_beta
      gradiente <- -apply(t(cbind(1,x)) %*% e, 1, sum)
      names(gradiente)[1] <- 'Intercept'
      gradiente
    }
   grad_fun
}


descenso <- function(n, z_0, eta, h_grad){
  # esta función calcula n iteraciones de descenso en gradiente 
  z <- matrix(0,n, length(z_0))
  z[1, ] <- z_0
  for(i in 1:(n-1)){
    z[i+1,] <- z[i,] - eta*h_grad(z[i,])
  }
  z
}



## 2. Calcula las funciones para nuestro problema:
x <- dat_e_norm %>% select(-id, -grasacorp)
y <- dat_e_norm$grasacorp

rss <- rss_calc(x, y)
grad <- grad_calc(x, y) 

## 3. Haz descenso en gradiente

#define z_0 y eta
z_0 <- rep(0, 14) 
eta <- 0.001
n <- 2000
z <- descenso(n, z_0, eta, grad)

## grafica evolución de rss
plot(apply(z, 1, rss))

#define betas finales
beta <- z[n,]
#calcula error de entrenamiento
sqrt(rss(beta)/nrow(dat_e_norm))
beta
## 4. Evalúa con muestra de prueba (observa que es el mismo código de arriba
## para normalizar)

dat_p_norm <- dat_p %>%
  gather(variable, valor, edad:muñeca) %>%
  left_join(normalizacion) %>%
  mutate(valor_norm = (valor - media)/de) %>%
  select(id, grasacorp, variable, valor_norm) %>%
  spread(variable, valor_norm) 


y_p <- dat_p_norm$grasacorp
x_p <- dat_p_norm %>% select(-id, -grasacorp)
rss_prueba <- rss_calc(x_p, y_p)

# raíz de error cuadrático medio
sqrt(rss_prueba(beta)/nrow(dat_p_norm))

# grafica predicciones contra observados
dat_p_norm$pred <- as.matrix(cbind(1, x_p)) %*% beta
ggplot(dat_p_norm, aes(x=pred, y = grasacorp)) + geom_point() +
  geom_abline(slope=1, intercept=0)


### 5. Compara con lm
mod_lineal <- lm(grasacorp ~ ., data = dat_e_norm %>% select(-id))
coefficients(mod_lineal)
