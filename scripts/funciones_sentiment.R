library(tidytext)
library(SnowballC) #para hacer stemming

### Leer datos y preproceso básico

leer_datos <- function(tipo, path){
  lista_archivos <- list.files(path=paste0(path, tipo), 
                               pattern=".txt", full.names = T)
  lista_textos <- lapply(lista_archivos, function(archivo){
    con <- file(archivo, "r", blocking = FALSE)
    lineas <- readLines(con = con)
    close(con)
    lineas[1] <- paste('Review ',lineas[1], sep=' ')
    lineas
  })
  lapply(lista_textos, function(x){
    y <- paste(x, collapse=' ')  
    y
  })
}
prep_df <- function(path){
  negativas_lista <- leer_datos('neg', path)
  positivas_lista <- leer_datos('pos', path)
  df_neg <- data_frame(id =1:1000, tipo ='neg', texto = negativas_lista)
  df_pos <- data_frame(id =1001:2000, tipo ='pos', texto = positivas_lista)
  df <- bind_rows(df_pos, df_neg)
}


## Tokenizar

calc_vocabulario <- function(df_ent, n = 100, remove_stop = FALSE, stem=FALSE,
                             bigram=FALSE){
  ## Extrae los n términos más comunes de 
  ## df_ent, que es una tabla con columna "texto"
  if(!bigram){
  df_tokens_ent <- df_ent %>%
    unnest_tokens(palabra, texto, token = 'words')
  } else {
    df_tokens_ent <- df_ent %>%
      unnest_tokens(palabra, texto, token = 'ngrams', n=2)
  }
  if(remove_stop){
    stops <- stop_words %>% rename(palabra = word)
    df_tokens_ent <- df_tokens_ent %>% anti_join(stops, by ='palabra')
  }
  if(stem){
    df_tokens_ent <- df_tokens_ent %>% mutate(palabra = wordStem(palabra))
  }
  frec_tokens <- df_tokens_ent %>% 
    group_by(palabra) %>% 
    summarise(frec = n()) 
  vocabulario <- top_n(frec_tokens, n, frec) 
  vocabulario
}

tokenizar_datos <- function(df, vocabulario, stem=FALSE, bigrams = FALSE){
  ## Calcula tabla con un renglón por término (token),
  ## donde los términos deben estar en la tabla vocabulario
  if(!bigrams){
    df_frecs <- df %>% unnest_tokens(palabra, texto, token = 'words')
  } else {
    df_frecs <- df %>% unnest_tokens(palabra, texto, token = 'ngrams', n=2)
  }
  if(stem){
    df_frecs <- df_frecs %>% mutate(palabra = wordStem(palabra))
  }
  df_frecs <- df_frecs %>%
    group_by(id, palabra, tipo) %>%
    summarise(frec_doc = n())
  df_frecs_filtrado <- full_join(df_frecs, vocabulario) %>% 
    select(-frec) %>%
    inner_join(vocabulario %>% select(palabra), by = c('palabra')) 
  df_frecs_filtrado
}



obtener_xy <- function(df){
  df <- df %>% arrange(id)
  x <- df %>% ungroup %>% 
    cast_sparse(id, palabra, frec_doc) 
  x <- x[!is.na(rownames(x)), ]
  y <-  df %>% ungroup %>% select(id, tipo) %>% unique
  y <- y[!is.na(y$id), ]
  #checar orden
  if(!all(as.numeric(rownames(x)) ==y$id)){
    stop('no coinciden indices de x y y')
  }
  list(x = x[, sort(colnames(x))], y = as.numeric(y$tipo=='pos'))
}


convertir_lista <- function(df, vocabulario, stem = FALSE){
  vocabulario <- arrange(vocabulario, desc(frec))
  vocabulario$id_palabra <- 2:(nrow(vocabulario)+1)
  df_filt <- df %>% unnest_tokens(palabra, texto, token = 'words') %>%
    ungroup %>%
    arrange(id) %>%
    full_join(vocabulario) %>% 
    select(-frec) %>%
    inner_join(vocabulario %>% select(id_palabra), by = c('id_palabra')) %>%
    filter(!is.na(id))
  y_df <- df_filt %>% select(id, tipo) %>% unique
  y_tipo <- y_df %>% pull(tipo)
  y <- as.numeric(y_tipo == 'pos')
  #y <- y[!is.na(y)]
  #df_filt$id_palabra <- as.integer(df_filt$id_palabra)
  lista <- split(df_filt$id_palabra, df_filt$id)
  out <- lapply(lista, function(elem) {c(1L, elem)})
  ids <- names(out)
  #checar ids
  if(!all(as.integer(ids)==y_df$id)){
    stop('Indices no concuerdan.')
  }
  names(out) <- NULL
  list(x=out,y=y)
  }


correr_modelo <- function(df_ent, df_pr, vocabulario, 
                          alpha = 0, lambda = NULL, stem=FALSE){
  df_filt_ent <- tokenizar_datos(df_ent, vocabulario, stem = stem)
  df_filt_pr <- tokenizar_datos(df_pr, vocabulario, stem = stem)
  mat_ent <- obtener_xy(df_filt_ent)
  dim(mat_ent$x)
  length(mat_ent$y)
  mat_pr <- obtener_xy(df_filt_pr)
  
  mod_reg <- glmnet(x = mat_ent$x , y = mat_ent$y , alpha = alpha, 
                       lambda =lambda, standardize = TRUE,
                       family ='binomial')
  #plot(mod_reg)
  preds <- predict(mod_reg, newx = mat_ent$x, type ='class', s = lambda)
  preds_pr <- predict(mod_reg, newx = mat_pr$x, type ='class', s=lambda)
  
  probs <- predict(mod_reg, newx = mat_ent$x, type ='link', s = lambda)
  probs_pr <- predict(mod_reg, newx = mat_pr$x, type ='link', s=lambda)
  dev_entrena <- round(-2*mean(mat_ent$y*probs - log(1+exp(probs)) ),3)
  dev_prueba <- round(-2*mean(mat_pr$y*probs_pr - log(1+exp(probs_pr)) ),3)
  #dev_entrena <- -2*mean(mat_ent$y*log(probs) + (1-mat_ent$y)*log(1-probs))
  #dev_prueba <- -2*mean(mat_pr$y*log(probs_pr) + (1-mat_pr$y)*log(1-probs_pr))
  #table(preds, mat_ent$y)
  #table(preds_pr, mat_pr$y)
  err_ent <- mean(preds!=mat_ent$y)
  err_prueba <- mean(preds_pr!=mat_pr$y)
  print(paste0('Error entrenamiento: ', round(err_ent,2)))
  print(paste0('Error prueba: ', round(err_prueba,2)))
  print(paste0('Devianza entrena:', dev_entrena ))
  print(paste0('Devianza prueba:', dev_prueba ))
  mod_reg
}


correr_modelo_cv <- function(df_ent, df_pr, vocabulario, 
                             alpha = 0, lambda = NULL, 
                             stem=FALSE, bigram = FALSE, 
                             standardize = TRUE){
  if(!bigram){
    df_filt_ent <- tokenizar_datos(df_ent, vocabulario, stem = stem)
    df_filt_pr <- tokenizar_datos(df_pr, vocabulario, stem = stem)
  } else {
    df_filt_ent <- tokenizar_datos(df_ent, vocabulario, bigram = TRUE)
    df_filt_pr <- tokenizar_datos(df_pr, vocabulario, bigram = TRUE)
  }
  mat_ent <- obtener_xy(df_filt_ent)
  dim(mat_ent$x)
  length(mat_ent$y)
  mat_pr <- obtener_xy(df_filt_pr)
  
  mod_reg <- cv.glmnet(x = mat_ent$x , y = mat_ent$y , alpha = alpha, 
                       lambda = lambda, standardize = standardize,
                    family ='binomial', parallel = TRUE)
  list(mod = mod_reg, entrena = mat_ent, prueba = mat_pr)
}

describir_modelo_cv <- function(corrida){
  #corrida es la salida de función correr_modelo_cv  
  mod_reg <- corrida$mod
  mat_ent <- corrida$entrena
  mat_pr <- corrida$prueba
  plot(mod_reg)
  preds <- predict(mod_reg, newx = mat_ent$x, type ='class', s = 'lambda.min')
  preds_pr <- predict(mod_reg, newx = mat_pr$x, type ='class', s= 'lambda.min')
  probs <- predict(mod_reg, newx = mat_ent$x, type ='link', s = 'lambda.min')
  probs_pr <- predict(mod_reg, newx = mat_pr$x, type ='link', s= 'lambda.min')
  dev_entrena <- round(-2*mean(mat_ent$y*probs - log(1+exp(probs)) ),3)
  dev_prueba <- round(-2*mean(mat_pr$y*probs_pr - log(1+exp(probs_pr)) ),3)
  #dev_entrena <- -2*mean(mat_ent$y*log(probs) + (1-mat_ent$y)*log(1-probs))
  #dev_prueba <- -2*mean(mat_pr$y*log(probs_pr) + (1-mat_pr$y)*log(1-probs_pr))
  #table(preds, mat_ent$y)
  #table(preds_pr, mat_pr$y)
  err_ent <- mean(preds!=mat_ent$y)
  err_prueba <- mean(preds_pr!=mat_pr$y)
  print(paste0("Lambda min: ", as.character(mod_reg$lambda.min)))
  print(paste0('Error entrenamiento: ', round(err_ent,2)))
  print(paste0('Error prueba: ', round(err_prueba,2)))
  print(paste0('Devianza entrena:', dev_entrena ))
  print(paste0('Devianza prueba:', dev_prueba ))
  #list(mod = mod_reg, entrena = mat_ent, prueba = mat_pr)
}


