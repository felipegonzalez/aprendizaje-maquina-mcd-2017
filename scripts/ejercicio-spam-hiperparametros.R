if(Sys.info()['nodename'] == 'vainilla.local'){
  # esto es por mi instalación particular de tensorflow - típicamente
  # no es necesario que corras esta línea.
  Sys.setenv(TENSORFLOW_PYTHON="/usr/local/bin/python")
}

library(keras)
library(readr)
library(tidyr)
library(dplyr)
spam_entrena <- read_csv('./datos/spam-entrena.csv') 
spam_prueba <- read_csv('./datos/spam-prueba.csv')
set.seed(293)
## ordenamos al azar
spam_entrena <- sample_n(spam_entrena, nrow(spam_entrena))

## Usaremos el último 20% para validar. Tenemos que
## hacer entonces el cálculo correctamente
spam_entrena$u <- 1:nrow(spam_entrena)/nrow(spam_entrena)

spam_entrena_1 <- spam_entrena %>% filter(u < 0.8)
spam_valida <- spam_entrena %>% filter(u >= 0.2)

x_ent <- spam_entrena_1 %>% select(-X1, -spam, -u) %>% as.matrix
x_valid <- spam_valida %>% select(-X1, -spam, -u) %>% as.matrix
x_ent_s <- scale(x_ent)
media <- attr(x_ent_s, 'scaled:center')
sd <- attr(x_ent_s, 'scaled:scale')
x_valid_s <- scale(x_valid, center = media, scale = sd)

y_ent <- spam_entrena_1$spam
y_valid <- spam_valida$spam

#Los pasamos juntos a keras, que usa el último porcentaje para validación
x_ent_s <- rbind(x_ent_s, x_valid_s)
y_ent <- c(y_ent, y_valid)

#x_prueba <- spam_prueba %>% select(-X1, -spam) %>% as.matrix 
#x_prueba_s <- x_valid %>%
#  scale(center = attr(x_ent_s, 'scaled:center'), scale = attr(x_ent_s,  'scaled:scale'))

#y_prueba <- spam_prueba$spam


correr_modelo <- function(params, x_ent_s, y_ent,  valid_prop=0.2){
  modelo_tc <- keras_model_sequential() 
  u <- params[['init_pesos']]
  modelo_tc %>% 
    layer_dense(units = params[['n_capa']], activation = 'sigmoid', 
                kernel_regularizer = regularizer_l2(l = params[['lambda']]), 
                kernel_initializer = initializer_random_uniform(minval = -u, 
                                                                maxval = u),
                input_shape=57) %>% 
    layer_dense(units = 1, activation = 'sigmoid',
                kernel_regularizer = regularizer_l2(l = params[['lambda']]),
                kernel_initializer = initializer_random_uniform(minval = -u, 
                                                                maxval = u)) 
  modelo_tc %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_sgd(lr =params[['lr']]),
    metrics = c('accuracy', 'binary_crossentropy')
  )
  history <- modelo_tc %>% fit(
    x_ent_s, y_ent, 
    epochs = params[['n_iter']], batch_size = nrow(x_ent_s), 
    verbose = 1,
    callbacks = callback_tensorboard(paste0("logs_spam/", params[['nombre']])),
    validation_split = valid_prop,
    shuffle = FALSE)
  list(history, modelo_tc)
}

params <- list(init_pesos = 0.5,
               lambda = 0.0001,
               n_capa = 50, # similar al número de variables
               lr = 0.1,
               n_iter = 100,
               nombre = 'corrida_1'
               )
iteraciones <- correr_modelo(params, x_ent_s, y_ent)

tensorboard('logs_spam')


# Podemos probar una tasa de aprendizaje más alta y más iteraciones
params <- list(init_pesos = 0.5,
               lambda = 0.0001,
               n_capa = 50,
               lr = 0.6,
               n_iter = 200,
               nombre = 'corrida_2')
iteraciones <- correr_modelo(params, x_ent_s, y_ent)

# Podemos probar una tasa de aprendizaje más alta 
params <- list(init_pesos = 0.5,
               lambda = 0.0001,
               n_capa = 50,
               lr = 0.8,
               n_iter = 200,
               nombre = 'corrida_2_1')
iteraciones <- correr_modelo(params, x_ent_s, y_ent)

# VEmos algunas oscilaciones, así que no vamos a incrementar más
# la tasa de aprendizaje.
#El error de entrenamiento está bastante cerca de prueba. Probemos intentar
#bajar los dos, reduciendo regularización. 

params <- list(init_pesos = 0.5,
               lambda = 0.00001,
               n_capa = 50,
               lr = 0.8,
               n_iter = 200,
               nombre = 'corrida_3')
iteraciones <- correr_modelo(params, x_ent_s, y_ent)

###  Ahora intentamos incrementar también el número de unidades
params <- list(init_pesos = 0.5,
               lambda = 0.00001,
               n_capa = 150,
               lr = 0.8,
               n_iter = 200,
               nombre = 'corrida_4_2')

iteraciones <- correr_modelo(params, x_ent_s, y_ent)

###  Pero ecesitamos reducir la tasa de aprendizaje
params <- list(init_pesos = 0.5,
               lambda = 0.00001,
               n_capa = 150,
               lr = 0.6,
               n_iter = 200,
               nombre = 'corrida_4_3')

iteraciones <- correr_modelo(params, x_ent_s, y_ent)

#Este camino no nos llevó a mejoras. Regresamos, intentando disminuir
# aún mas la regularización
params <- list(init_pesos = 0.5,
               lambda = 0.000001,
               n_capa = 50,
               lr = 0.8,
               n_iter = 200, 
               nombre = 'corrida_5')
iteraciones <- correr_modelo(params, x_ent_s, y_ent)



#Regresamos el valor de lambda. Intentamos correr por más tiempo, 
#con 100 unidades
params <- list(init_pesos = 0.5,
               lambda = 0.00001,
               n_capa = 100,
               lr = 0.8,
               n_iter = 500, 
               nombre = 'corrida_6_1')
iteraciones <- correr_modelo(params, x_ent_s, y_ent)

# Y hacemos una corrida con más iteraciones, 75 unidades
params <- list(init_pesos = 0.5,
               lambda = 0.00001,
               n_capa = 100,
               lr = 0.8,
               n_iter = 800, 
               nombre = 'corrida_7')
iteraciones <- correr_modelo(params, x_ent_s, y_ent)

##Y hacemos una corrida final con todos los datos
## Probamos nuestro modelo
params['nombre'] <- 'corrida_8'
params['n_iter'] <- 1000


x_ent <- spam_entrena %>% select(-X1, -spam, -u) %>% as.matrix
x_prueba <- spam_prueba %>% select(-X1, -spam) %>% as.matrix
x_ent_s <- scale(x_ent)
media <- attr(x_ent_s, 'scaled:center')
sd <- attr(x_ent_s, 'scaled:scale')
x_prueba_s <- scale(x_prueba, center = media, scale = sd)

y_ent <- spam_entrena$spam
y_prueba <- spam_prueba$spam



iteraciones <- correr_modelo(params, x_ent_s, y_ent, valid_prop = 0)

score <- iteraciones[[2]] %>% evaluate(x_prueba_s, y_prueba)
score
tab_confusion <- table(iteraciones[[2]] %>% 
                         predict_classes(x_prueba_s), y_prueba) 
tab_confusion
prop.table(tab_confusion, 2)
