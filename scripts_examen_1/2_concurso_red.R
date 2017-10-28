library(tidyverse)
if(Sys.info()['nodename'] == 'vainilla.local'){
  # esto es por mi instalación particular de tensorflow - típicamente
  # no es necesario que corras esta línea.
  Sys.setenv(TENSORFLOW_PYTHON="/usr/local/bin/python")
}
library(keras)
use_session_with_seed(42172)

########## Preparación de datos

if(FALSE){
  entrena <- read_csv('./concurso/train.csv', progress=FALSE)
  prueba <- read_csv('./concurso/test.csv', progress=FALSE)
  saveRDS(entrena, './concurso/entrena_imagenes.rds')
  saveRDS(prueba, './concurso/prueba_imagenes.rds')
}
entrena <- readRDS('./concurso/entrena_imagenes.rds')
prueba <-  readRDS('./concurso/prueba_imagenes.rds')

# Esto normalmente no está disponible
solucion_d <- read_csv('./concurso/derived.csv')
index_pub <- solucion_d$Usage=='Public'
index_private <- solucion_d$Usage=='Private'
sol_pub <- filter(solucion_d, Usage == 'Public')
sol_private <- filter(solucion_d, Usage == 'Private')


set.seed(3434)
x_entrena <- entrena %>% select(-estado, -hora) %>% as.matrix 
x_prueba <- prueba %>% select(-hora, -id) %>% as.matrix 

###### Crear cortes por segmentos (como en el script de regresión para validación
###### cruzada)

library(lubridate)
entrena_seg <- entrena %>% select(hora, estado) %>% mutate(fecha_hora = ymd_hms(hora)) %>%
  mutate(diferencia = fecha_hora - lag(fecha_hora, default = 0))

entrena_seg <- entrena_seg %>% mutate(dif_grande = diferencia > 9700) %>%
  mutate(segmento = cumsum(dif_grande))
entrena_seg %>% select(fecha_hora, diferencia, segmento) %>% print(n = 30)
max(entrena_seg$segmento)
# distribución de número de fotografías por segmento
entrena_seg %>% group_by(segmento) %>% tally %>% pull(n) %>% table

entrena_seg <- entrena_seg %>% mutate(foldid = (segmento %/% 2)+1)
table(entrena_seg$foldid)

#### Preparar datos para red

x_entrena_red <- x_entrena/255
dim(x_entrena_red) <- c(400, 200, 190,1)
x_prueba_red <- x_prueba/255
dim(x_prueba_red) <- c(588, 200, 190,1)
input_shape <- c(200, 190, 1)

y_entrena <- as.numeric(entrena$estado == 'cerrada')

### Para ver como escogimos este modelo, ver código de abajo
set.seed(83679)
model <- keras_model_sequential()
model %>% 
  layer_reshape(input_shape=input_shape, target_shape=input_shape) %>%
  layer_cropping_2d(cropping = list(list(80,20), list(20,20))) %>%
  layer_conv_2d(filters=8, kernel_size=c(3,3)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.4) %>%
  layer_conv_2d(filters=8, kernel_size=c(3,3)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_flatten() %>%
  layer_dropout(0.4) %>%
  layer_dense(7, activation='relu') %>%
  layer_dropout(0.4) %>%
  layer_dense(1, activation ='sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.006, momentum=0.7, decay = 1e-7),
  metrics = c('accuracy', 'binary_crossentropy'))

historia <- model %>% fit(
  x_entrena_red[entrena_seg$foldid <= 8,,,,drop=FALSE],
  y_entrena[entrena_seg$foldid <= 8],
  batch_size = 50,
  epochs = 100,
  verbose = 1, shuffle = TRUE,
  validation_data = list(x_entrena_red[entrena_seg$foldid >8,,,,drop=FALSE],
                       y_entrena[entrena_seg$foldid > 8])  )

# El desempeño es comparable a regresión, pero no suficiente
# para ganar el concurso. Prueba usando más datos (por ejemplo, con los
# datos completos, intenta unos 700-800 de entrenamiento)
#Public
evaluate(model, x_prueba_red[index_pub,,,,drop=FALSE], solucion_d$estado[index_pub])
#Private
evaluate(model, x_prueba_red[index_private,,,,drop=FALSE], solucion_d$estado[index_private])



###########################################################

### En lugar de usar validación cruzada tomamos una muestra
### de validación - estas redes tardan más en ajustarse. Puedes intentar
### también con redes más simples.
set.seed(1230)
seeds <- sample(1:100023, 50)
res <- lapply(1:50, function(i){
  set.seed(seeds[i])
  params <- list(
  filters = sample(4:12, 1),
  dropout_1 = runif(1, 0.3, 0.75),
  dropout_2 = runif(1, 0.3,0.75),
  dropout_3 = runif(1, 0.3,0.75),
  lr = exp(runif(1, -6, -4)),
  momentum = runif(1,0,0.7),
  units = sample(5:20, 1)
  )
  
model <- keras_model_sequential()

model %>% 
  layer_reshape(input_shape=input_shape, target_shape=input_shape) %>%
  layer_cropping_2d(cropping = list(list(80,20), list(20,20))) %>%
  layer_conv_2d(filters=params$filters, kernel_size=c(3,3)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(params$dropout_1) %>%
  layer_conv_2d(filters=params$filters, kernel_size=c(3,3)) %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_flatten() %>%
  layer_dropout(params$dropout_2) %>%
  layer_dense(params$units, activation='relu') %>%
  layer_dropout(params$dropout_3) %>%
  layer_dense(1, activation ='sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=params$lr, momentum=params$momentum, decay = 1e-7),
  metrics = c('accuracy', 'binary_crossentropy'))

model_completo <- clone_model(model)

historia <- model %>% fit(
  x_entrena_red[entrena_seg$foldid <= 8,,,,drop=FALSE],
  y_entrena[entrena_seg$foldid <= 8],
  batch_size = 50,
  epochs = 200,
  verbose = 0, shuffle = TRUE)
  #validation_data = list(x_entrena_red[entrena_seg$foldid >5,,,,drop=FALSE],
  #                       y_entrena[entrena_seg$foldid > 5])  )
#Entrena
eval_t <- evaluate(model, x_entrena_red, y_entrena)
#Valida
eval_v <- evaluate(model, x_entrena_red[entrena_seg$foldid >8,,,,drop=FALSE],
         y_entrena[entrena_seg$foldid > 8])$binary_crossentropy
out <- list(train=eval_t, eval=eval_v, params = params, 
            model = serialize_model(model))
print(out)
out
})

saveRDS(res, './cache_obj/res_200_fold8.rds')

indice <- which.min(as.numeric(sapply(res, function(elem) elem$eval)))
modelo <- res[[indice]]$model








