# Reducción de dimensionalidad

En esta parte veremos métodos no supervisados: no existe variable a predecir.
En estos métodos buscamos reexpresiones útiles de nuestros datos para que
sean más fáciles de entender o de explorar, para intentar comprimirlos, para
poder usarlos de manera más conveniente en procesos posteriores (por ejemplo,
predicción).

En esta parte veremos técnicas de reducción de dimensionalidad, 
en particular la descomposición en valores singulares, que es una de las más útiles.
La descomposición en valores singulares puede verse como un tipo de
descomposición aditiva en matrices de rango uno, así que comenzaremos
explicando estos conceptos.


## Descomposición aditiva en matrices de rango 1


Supongamos que tenemos una matriz de datos $X$ de tamaño $n\times p$ (todas
son variables numéricas). En los renglones tenemos los casos ($n$) y
las columnas son las variables $p$. 
Típicamente pensamos que las columnas o variables están
todas definidas en una misma escala:
por ejemplo, cantidades de dinero, poblaciones, número de eventos, etc. Cuando no
es así, entonces normalizamos las variables de alguna forma para no tener unidades.

### Matrices de rango 1

Una de las estructuras de datos más simples que podemos imaginar (que sea interesante)
para un tabla de este tipo es que se trata de una matriz de datos de rango 1.
Es  generada por un *score* de individuos que determina mediante un *peso* el valor
de una variable. Es decir, el individuo $i$ en la variable $j$ es

$$X_{ij} = \sigma u_i v_j$$

Donde $u=(u_1,u_2,\ldots, u_n)$ son los *scores* de los individuos y 
$v = (v_1, v_2, \ldots, v_p)$ son los pesos de las variables. Tanto
$u$ como $v$ son vectores normalizados, es decir $||u||=||v||=1$. La constante
$\sigma$ nos permite pensar que los vectores $u$ y $v$ están normalizados.


Esto se puede escribir, en notación matricial, como

$$X = \sigma u v^t$$
donde consideramos a $u$ y $v$ como matrices columna.

\BeginKnitrBlock{comentario}<div class="comentario">
- Una matriz de rango uno (o en general de rango bajo) es más simple de analizar.
En rango 1, tenemos que entender la variación de $n+p$ datos (componentes de $u$ y $v$),
mientras que en una matriz general tenemos que entender $n\times p$ datos.

- Cada variable $j$ de las observaciones es un reescalamiento del índice o score $u$
  de las personas por el factor $\sigma v_j$. 
Igualmente, cada caso $i$ de las observaciones es un reescalamiento del índice o peso
$v$ de las variables por el factor $\sigma u_i$.

- $u$ y $v$ representan una dimensión (dimensión latente, componente) de estos datos.</div>\EndKnitrBlock{comentario}


### Ejemplo: una matriz de rango 1 de preferencias {-}

Supongamos que las columnas de $X$ son películas ($p$), los renglones ($n$)
personas, y la entrada $X_{ij}$ es la afinidad de la persona $i$ por la película $j$.
Vamos a suponer que estos datos tienen una estructura ficticia de rango 1, basada
en las preferencias de las personas por películas de ciencia ficción.

Construimos los pesos de las películas que refleja qué tanto son de 
ciencia ficción o no.  Podemos pensar que cada uno de estos valores el el *peso* de la película en 
la dimensión de ciencia ficción.



```r
library(tidyverse)
peliculas_nom <- c('Gladiator','Memento','X-Men','Scream','Amores Perros',
               'Billy Elliot', 'Lord of the Rings','Mulholland drive',
                'Amelie','Planet of the Apes')
# variable latente que describe el contenido de ciencia ficción de cada 
v <- c(-1.5, -0.5, 4, -1,-3,  -3, 0, 1, -0.5, 3.5)
normalizar <- function(x){
  norma <- sqrt(sum(x^2))
  if(norma > 0){
    x_norm <- x/norma
  } else {
    x_norm <- x
  }
  x_norm
}
v <- normalizar(v)
peliculas <- data_frame(pelicula = peliculas_nom, v = v) %>% arrange(v)
peliculas
```

```
## # A tibble: 10 x 2
##              pelicula         v
##                 <chr>     <dbl>
##  1      Amores Perros -0.420084
##  2       Billy Elliot -0.420084
##  3          Gladiator -0.210042
##  4             Scream -0.140028
##  5            Memento -0.070014
##  6             Amelie -0.070014
##  7  Lord of the Rings  0.000000
##  8   Mulholland drive  0.140028
##  9 Planet of the Apes  0.490098
## 10              X-Men  0.560112
```

Ahora pensamos que tenemos con individuos con *scores* de qué tanto les gusta
la ciencia ficción


```r
set.seed(102)
u <- rnorm(15, 0, 1)
u <- normalizar(u)
personas <- data_frame(persona = 1:15, u = u)
head(personas)
```

```
## # A tibble: 6 x 2
##   persona           u
##     <int>       <dbl>
## 1       1  0.04215632
## 2       2  0.18325375
## 3       3 -0.31599557
## 4       4  0.46314650
## 5       5  0.28921210
## 6       6  0.28037223
```

Podemos entonces construir la afinidad de cada persona por cada película (matriz
$n\times p$ ) multiplicando
el *score* de cada persona (en la dimensión ciencia ficción) por el
peso de la película (en la dimensión ciencia ficción). Por ejemplo, para
una persona, tenemos que su índice es


```r
personas$u[2]
```

```
## [1] 0.1832537
```

Esta persona tiene afinidad por la ciencia ficción, así que sus niveles de gusto
por las películas son (multiplicando por $\sigma = 100$, que en este caso es
una constante arbitraria seleccionada para el ejemplo):

```r
data_frame(pelicula=peliculas$pelicula,  afinidad= 100*personas$u[2]*peliculas$v) %>%
  arrange(desc(afinidad))
```

```
## # A tibble: 10 x 2
##              pelicula  afinidad
##                 <chr>     <dbl>
##  1              X-Men 10.264263
##  2 Planet of the Apes  8.981230
##  3   Mulholland drive  2.566066
##  4  Lord of the Rings  0.000000
##  5            Memento -1.283033
##  6             Amelie -1.283033
##  7             Scream -2.566066
##  8          Gladiator -3.849099
##  9      Amores Perros -7.698197
## 10       Billy Elliot -7.698197
```

Consideremos otra persona


```r
personas$u[15]
```

```
## [1] -0.05320133
```

Esta persona tiene disgusto ligero por la ciencia ficción, y sus scores
de las películas son:


```r
data_frame(pelicula=peliculas$pelicula,  afinidad= 100*personas$u[15]*peliculas$v)
```

```
## # A tibble: 10 x 2
##              pelicula   afinidad
##                 <chr>      <dbl>
##  1      Amores Perros  2.2349029
##  2       Billy Elliot  2.2349029
##  3          Gladiator  1.1174515
##  4             Scream  0.7449676
##  5            Memento  0.3724838
##  6             Amelie  0.3724838
##  7  Lord of the Rings  0.0000000
##  8   Mulholland drive -0.7449676
##  9 Planet of the Apes -2.6073868
## 10              X-Men -2.9798706
```

Si fuera tan simple el gusto por las películas (simplemente depende si contienen
ciencia ficción o no, y si a la persona le gusta o no), la matriz $X$ de observaciones
sería

$$X_1 = \sigma uv^t$$
donde consideramos a $u$ y $v$ como vectores columna. El producto es
de una matriz de $n\times 1$ contra una de $1\times p$, lo cual da una matriz de $n\times p$.


Podemos calcular como:


```r
X = 100*tcrossprod(personas$u, peliculas$v ) # tcrossprod(x,y) da x %*% t(y)
colnames(X) <- peliculas$pelicula
head(round(X, 1))
```

```
##      Amores Perros Billy Elliot Gladiator Scream Memento Amelie
## [1,]          -1.8         -1.8      -0.9   -0.6    -0.3   -0.3
## [2,]          -7.7         -7.7      -3.8   -2.6    -1.3   -1.3
## [3,]          13.3         13.3       6.6    4.4     2.2    2.2
## [4,]         -19.5        -19.5      -9.7   -6.5    -3.2   -3.2
## [5,]         -12.1        -12.1      -6.1   -4.0    -2.0   -2.0
## [6,]         -11.8        -11.8      -5.9   -3.9    -2.0   -2.0
##      Lord of the Rings Mulholland drive Planet of the Apes X-Men
## [1,]                 0              0.6                2.1   2.4
## [2,]                 0              2.6                9.0  10.3
## [3,]                 0             -4.4              -15.5 -17.7
## [4,]                 0              6.5               22.7  25.9
## [5,]                 0              4.0               14.2  16.2
## [6,]                 0              3.9               13.7  15.7
```

O usando data.frames como


```r
peliculas %>% crossing(personas) %>% mutate(afinidad = round(100*u*v, 2)) %>%
  select(persona, pelicula, afinidad) %>%
  spread(pelicula, afinidad) 
```

```
## # A tibble: 15 x 11
##    persona Amelie `Amores Perros` `Billy Elliot` Gladiator
##  *   <int>  <dbl>           <dbl>          <dbl>     <dbl>
##  1       1  -0.30           -1.77          -1.77     -0.89
##  2       2  -1.28           -7.70          -7.70     -3.85
##  3       3   2.21           13.27          13.27      6.64
##  4       4  -3.24          -19.46         -19.46     -9.73
##  5       5  -2.02          -12.15         -12.15     -6.07
##  6       6  -1.96          -11.78         -11.78     -5.89
##  7       7  -1.47           -8.79          -8.79     -4.40
##  8       8  -0.41           -2.49          -2.49     -1.24
##  9       9  -0.90           -5.39          -5.39     -2.70
## 10      10  -3.11          -18.67         -18.67     -9.34
## 11      11  -2.36          -14.17         -14.17     -7.09
## 12      12  -0.19           -1.13          -1.13     -0.57
## 13      13   1.03            6.15           6.15      3.08
## 14      14   2.08           12.45          12.45      6.23
## 15      15   0.37            2.23           2.23      1.12
## # ... with 6 more variables: `Lord of the Rings` <dbl>, Memento <dbl>,
## #   `Mulholland drive` <dbl>, `Planet of the Apes` <dbl>, Scream <dbl>,
## #   `X-Men` <dbl>
```


Nótese que en este ejemplo podemos simplificar mucho el análisis: en lugar de ver
la tabla completa, podemos simplemente considerar los dos vectores de índices (pesos
 y scores), y trabajar como si fuera un problema de una sola dimensión.

---

## Aproximación con matrices de rango 1.

En general, las matrices de datos reales no son de rango 1. Más bien nos interesa
saber si se puede hacer una buena aproximación de rango 1.


\BeginKnitrBlock{comentario}<div class="comentario">El problema que nos interesa es el inverso: si tenemos la tabla $X$, ¿cómo sabemos
si se puede escribir aproximadamente en la forma simple de una matriz de rango uno? 
Nótese que si lo
pudiéramos hacer, esto simplificaría mucho nuestro análisis de estos datos, y 
obtendríamos información valiosa.</div>\EndKnitrBlock{comentario}


Medimos la diferencia entre una matriz de datos general $X$ y una matriz
de rango 1 $\sigma uv^t$ mediante la norma Frobenius:

$$ ||X-\sigma uv^t||^2_F = \sum_{i,j} (X_{i,j} - \sigma u_iv_j)^2$$

Nos interesa resolver

$$\min_{\sigma, u,v} || X - \sigma uv^t ||_F^2$$

donde $\sigma$ es un escalar, $u$ es un vector columna de tamaño $n$ y $v$ es un
vector columna de tamaño $p$. Suponemos que los vectores $u$ y $v$ tienen norma uno.
Esto no es necesario - podemos absorber constantes en $\sigma$.


### Ejemplo {-}

Por ejemplo, la siguiente tabla tiene gastos personales en distintos rubros en distintos 
años para todo Estados Unidos (en dólares nominales).



```r
library(tidyverse)
X_arr <- USPersonalExpenditure[, c(1,3,5)]
X_arr
```

```
##                       1940  1950  1960
## Food and Tobacco    22.200 59.60 86.80
## Household Operation 10.500 29.00 46.20
## Medical and Health   3.530  9.71 21.10
## Personal Care        1.040  2.45  5.40
## Private Education    0.341  1.80  3.64
```

En este ejemplo  podríamos tener la intuición de que la proporción de gasto
se ha mantenido aproximadamente constante en cada año, y que todos los rubros
han aumentado debido a la inflación. Podríamos intentar hacer
varias normalizaciones para probar esta idea, pero quisiéramos idear una
estrategia general.

Digamos que el vector $u$ denota los niveles generales de cada rubro (es un vector
de longitud 5), y el vector $v$ denota los niveles generales de cada año
(un vector de longitud 3). Queremos ver si es razonable aproximar
$$X\approx uv^t$$

**Observación**: En este caso, la ecuación de arriba $X_{i,j} = u_iv_j$  expresa
que hay niveles generales para cada rubro $i$ a lo largo de todos los años, y para obtener
una aproximación ajustamos con un factor $v_j$ de inflación el año $j$

La mejor manera de entender este problema es con álgebra lineal, como veremos más adelante.
Por el momento intentemos aproximar directamente, intentando resolver (podemos normalizar
$u$ y $v$ más tarde y encontrar la $\sigma$):

$$\min_{u,v} \sum_{i,j} (X_{i,j} - u_iv_j)^2 = \min_{u,v} ||X-uv^t||^2_F$$

**Observación**:Este problema tiene varios mínimos, pues podemos mover constantes de $u$
a $v$ (tiene múltiples soluciones). Hay varias maneras de lidiar con esto (por ejemplo,
normalizando). Por el momento, corremos la optimización para encontrar una solución:


```r
error <- function(pars){
  v <- pars[1:3]
  u <- pars[4:8]
  mean((X_arr - tcrossprod(u, v))^2) #tcrossprod da x %*% t(y)
}
optim_decomp <- optim(rep(0.1, 5 + 3), error, method ='BFGS')
```


```r
v_años <- optim_decomp$par[1:3]
u_rubros <- optim_decomp$par[4:8]
```


La matriz $X_1=uv^t$ que obtuvimos es:


```r
X_1 <- tcrossprod(u_rubros, v_años)
round(X_1, 1)
```

```
##      [,1] [,2] [,3]
## [1,] 21.6 58.4 87.8
## [2,] 11.1 30.1 45.3
## [3,]  4.7 12.6 18.9
## [4,]  1.2  3.2  4.8
## [5,]  0.8  2.2  3.3
```

Podemos ver qué tan buena es la aproximación:


```r
R <- X_arr - X_1
qplot(as.numeric(X_1), as.numeric(as.matrix(X_arr))) + geom_abline(colour='red')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-15-1.png" width="672" />

```r
round(R,2)
```

```
##                      1940  1950  1960
## Food and Tobacco     0.60  1.25 -0.98
## Household Operation -0.65 -1.11  0.90
## Medical and Health  -1.12 -2.87  2.18
## Personal Care       -0.15 -0.77  0.55
## Private Education   -0.46 -0.38  0.36
```


donde vemos que nuestra aproximación explica en buena parte la variación
de los datos en la tabla $X$. La descomposición que obtuvimos es de la forma
$$X = uv^t + R$$
donde $R$ tiene norma Frobenius relativamente chica.


**Observaciones**: 

- Este método nos da un ordenamiento de rubros de gasto según su nivel general, y
un ordenamiento de años según su nivel general de gasto.


```r
data_frame(rubro = rownames(X_arr), nivel = u_rubros) %>% arrange(desc(nivel))
```

```
## # A tibble: 5 x 2
##                 rubro     nivel
##                 <chr>     <dbl>
## 1    Food and Tobacco 9.7404116
## 2 Household Operation 5.0268356
## 3  Medical and Health 2.0992569
## 4       Personal Care 0.5380014
## 5   Private Education 0.3634288
```

```r
data_frame(año = colnames(X_arr), nivel = v_años)
```

```
## # A tibble: 3 x 2
##     año    nivel
##   <chr>    <dbl>
## 1  1940 2.217377
## 2  1950 5.990600
## 3  1960 9.011784
```

- Pudimos explicar estos datos usando esos dos índices (5+3=7 números) en lugar
de toda la tabla(5(3)=15 números).

- Una vez explicado esto, podemos concentrarnos en los patrones que hemos aislado
en la matriz $R$. Podríamos repetir buscando una aproximación igual a la que acabomos 
de hacer para la matriz $X$, o podríamos hacer distintos tipos de análisis.

---


### Suma de matrices de rango 1.

La matriz de datos $X$ muchas veces no puede aproximarse bien con una sola matriz
de rango 1. Podríamos entonces buscar descomponer los datos en más de una dimensión
latente:

$$X = \sigma_1 u_1v_1^t + \sigma_2 u_2v_2^t+\ldots+ \sigma_k u_kv_k^t$$

### Ejemplo: películas {-}

En nuestro ejemplo anterior, claramente debe haber otras dimensiones latentes
que expliquen la afinidad por una película. Por ejemplo, quizá podríamos considerar
el gusto por películas *mainstream* vs películas independientes.


```r
peliculas_nom <- c('Gladiator','Memento','X-Men','Scream','Amores Perros',
               'Billy Elliot', 'Lord of the Rings','Mulholland drive',
                'Amelie','Planet of the Apes')
# variable latente que describe el contenido de ciencia ficción de cada 
v_1 <- c(-1.5, -0.5, 4, -1,-3,  -3, 0, 1, -0.5, 3.5)
v_2 <- c(4.1, 0.2, 3.5, 1.5, -3.0, -2.5, 2.0, -4.5, -1.0, 2.6) #mainstream o no
v_1 <- normalizar(v_1)
v_2 <- normalizar(v_2)
peliculas <- data_frame(pelicula = peliculas_nom, v_1 = v_1, v_2 = v_2) %>% arrange(v_2)
peliculas
```

```
## # A tibble: 10 x 3
##              pelicula       v_1         v_2
##                 <chr>     <dbl>       <dbl>
##  1   Mulholland drive  0.140028 -0.50754390
##  2      Amores Perros -0.420084 -0.33836260
##  3       Billy Elliot -0.420084 -0.28196884
##  4             Amelie -0.070014 -0.11278753
##  5            Memento -0.070014  0.02255751
##  6             Scream -0.140028  0.16918130
##  7  Lord of the Rings  0.000000  0.22557507
##  8 Planet of the Apes  0.490098  0.29324759
##  9              X-Men  0.560112  0.39475637
## 10          Gladiator -0.210042  0.46242889
```

Y las personas tienen también *scores* en esta nueva dimensión, que aquí simulamos al azar


```r
personas <- personas %>% mutate(u_1 = u, u_2 = normalizar(rnorm(15, 0, 1))) %>% select(-u)
head(personas)
```

```
## # A tibble: 6 x 3
##   persona         u_1         u_2
##     <int>       <dbl>       <dbl>
## 1       1  0.04215632 -0.06231047
## 2       2  0.18325375 -0.37654521
## 3       3 -0.31599557  0.41966960
## 4       4  0.46314650 -0.40146285
## 5       5  0.28921210  0.12705318
## 6       6  0.28037223  0.18002499
```
Por ejemplo, la segunda persona persona le gusta la  ciencia ficción,
y pero prefiere fuertemente películas  independientes.

Podemos graficar a las personas según su interés en ciencia ficción y mainstream:


```r
ggplot(personas, aes(x = u_1, y=u_2)) + geom_point() +
  geom_vline(xintercept = 0, colour='red') + 
  geom_hline(yintercept = 0, colour='red') + xlab('Ciencia ficción')+
  ylab('Mainstream')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-19-1.png" width="480" />


Y también podemos graficar las películas


```r
ggplot(peliculas, aes(x = v_1, y=v_2, label = pelicula)) + geom_point() +
  geom_vline(xintercept = 0, colour='red') + 
  geom_hline(yintercept = 0, colour='red')+ xlab('Ciencia ficción')+
  ylab('Mainstream') + geom_text()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-20-1.png" width="480" />


¿Cómo calculariamos ahora la afinidad de una persona por una película? Necesitamos
calcular (dando el mismo peso a las dos dimensiones)
$$X_{i,j} = \sigma_1 u_{1,i} v_{1,j} + \sigma_2 u_{2,i} v_{2,j}$$

Usamos la notación $u_{k,i}$ para denotar la componente $i$ del vector $u_k$.

Antes pusimos $\sigma_1=100$. Supongamos que la siguiente componente es un poco
menos importante que la primera. Podriamos escoger $\sigma_2=70$, por ejemplo.

Podríamos hacer

```r
library(purrr)
library(stringr)
personas_larga <- personas %>% gather(dimension, u, u_1:u_2) %>%
  separate(dimension, c('x','dim'), '_') %>% select(-x)
head(personas_larga)
```

```
## # A tibble: 6 x 3
##   persona   dim           u
##     <int> <chr>       <dbl>
## 1       1     1  0.04215632
## 2       2     1  0.18325375
## 3       3     1 -0.31599557
## 4       4     1  0.46314650
## 5       5     1  0.28921210
## 6       6     1  0.28037223
```

```r
peliculas_larga <- peliculas %>% gather(dimension, v, v_1:v_2) %>%
  separate(dimension, c('x','dim'), '_') %>% select(-x)
head(peliculas_larga)
```

```
## # A tibble: 6 x 3
##           pelicula   dim         v
##              <chr> <chr>     <dbl>
## 1 Mulholland drive     1  0.140028
## 2    Amores Perros     1 -0.420084
## 3     Billy Elliot     1 -0.420084
## 4           Amelie     1 -0.070014
## 5          Memento     1 -0.070014
## 6           Scream     1 -0.140028
```

```r
sigma_df <- data_frame(dim = c('1','2'), sigma = c(100,70))
```



```r
df_dim <- personas_larga %>% left_join(peliculas_larga) %>%
                        left_join(sigma_df) %>%
                        mutate(afinidad = sigma*u*v)
```

```
## Joining, by = "dim"
## Joining, by = "dim"
```

```r
df_agg <- df_dim %>% group_by(persona, pelicula) %>%
  summarise(afinidad = round(sum(afinidad),2))
df_agg %>% spread(pelicula, afinidad)
```

```
## # A tibble: 15 x 11
## # Groups:   persona [15]
##    persona Amelie `Amores Perros` `Billy Elliot` Gladiator
##  *   <int>  <dbl>           <dbl>          <dbl>     <dbl>
##  1       1   0.20           -0.30          -0.54     -2.90
##  2       2   1.69            1.22          -0.27    -16.04
##  3       3  -1.10            3.33           4.99     20.22
##  4       4  -0.07           -9.95         -11.53    -22.72
##  5       5  -3.03          -15.16         -14.66     -1.96
##  6       6  -3.38          -16.04         -15.33     -0.06
##  7       7  -1.07           -7.60          -7.80     -6.02
##  8       8   2.08            4.99           3.74    -11.46
##  9       9   0.95            0.17          -0.76    -10.29
## 10      10  -2.44          -16.65         -16.99    -12.10
## 11      11  -2.01          -13.12         -13.30     -8.52
## 12      12  -0.34           -1.60          -1.52      0.07
## 13      13   0.70            5.17           5.33      4.42
## 14      14   0.01            6.27           7.30     14.68
## 15      15  -3.43           -9.17          -7.27     16.70
## # ... with 6 more variables: `Lord of the Rings` <dbl>, Memento <dbl>,
## #   `Mulholland drive` <dbl>, `Planet of the Apes` <dbl>, Scream <dbl>,
## #   `X-Men` <dbl>
```


**Observación**: Piensa qué harías si vieras esta tabla directamente, e imagina
cómo simplificaría la comprensión y análisis si conocieras las matrices de rango 1
con las que se construyó este ejemplo.


Consideremos la persona 2:


```r
filter(personas, persona==2)
```

```
## # A tibble: 1 x 3
##   persona       u_1        u_2
##     <int>     <dbl>      <dbl>
## 1       2 0.1832537 -0.3765452
```

Que tiene gusto por la ciencia ficción y le gustan películas independientes.
Sus afinidades son:

```r
filter(df_agg, persona==2) %>% arrange(desc(afinidad))
```

```
## # A tibble: 10 x 3
## # Groups:   persona [1]
##    persona           pelicula afinidad
##      <int>              <chr>    <dbl>
##  1       2   Mulholland drive    15.94
##  2       2             Amelie     1.69
##  3       2 Planet of the Apes     1.25
##  4       2      Amores Perros     1.22
##  5       2              X-Men    -0.14
##  6       2       Billy Elliot    -0.27
##  7       2            Memento    -1.88
##  8       2  Lord of the Rings    -5.95
##  9       2             Scream    -7.03
## 10       2          Gladiator   -16.04
```

Explicaríamos así esta descomposición: 

- Cada persona $i$ tiene un nivel de gusto por 
ciencia ficción ($u_{1,i}$) y otro nivel de gusto por películas independientes ($u_{2,i}$).
- Cada película $j$ tiene una calificación o peso en la dimensión de ciencia ficción ($v_{1,i}$)
y un peso en la dimensión de independiente  ($v_{2,i}$)

La afinidad de una persona $i$ por una película $j$ se calcula como

$$ \sigma_1 u_{1,i}v_{1,j}  + \sigma_2 u_{2,i}v_{2,j}$$


\BeginKnitrBlock{comentario}<div class="comentario">- Una matriz de rango 2 es una suma (o suma ponderada) de matrices de rango 1
- Las explicaciones de matrices de rango aplican para cada sumando (ver arriba)
- En este caso, hay dos dimensiones latentes que explican los datos: preferencia por independientes
y preferencia por ciencia ficción. En este ejemplo ficticio estas componentes explica del todo
a los datos.</div>\EndKnitrBlock{comentario}

---



## Aproximación con matrices de rango bajo

\BeginKnitrBlock{comentario}<div class="comentario">Nuestro problema generalmente es el inverso: si tenemos la matriz de datos $X$,
¿podemos encontrar un número bajo $k$ de dimensiones de forma que $X$ se escribe (o aproxima) como suma de
matrices de $k$ matrices rango 1? Lograr esto sería muy bueno, pues otra vez
simplificamos el análisis a solo un número de dimensiones $k$ (muy) menor a $p$,
el número de variables, sin perder mucha información (con buen grado de aproximación).

Adicionalmente, las dimensiones encontradas pueden mostrar  patrones interesantes que iluminan los datos,
esqpecialmente en términos de aquellas dimensiones que aportan mucho a la aproximación.</div>\EndKnitrBlock{comentario}


En general, buscamos encontrar una aproximación de la matriz $X$ mediante una
suma de matrices de rango 1

$$X \approx \sigma_1 u_1v_1^t + \sigma_2 v_2v_2^t+\ldots+ \sigma_k u_kv_k^t.$$

A esta aproximación le llamamos una *aproximación de rango* $k$. Hay muchas maneras de hacer esto, y probablemente la mayoría de ellas no
son muy interesantes. Podemos más concretamente preguntar,  ¿cuál es la mejor aproximación
de rango $k$ que hay?

$$\min_{X_k} || X - X_k ||_F^2$$
donde consideramos la distancia entre $X$ y $X_k$ con la norma de Frobenius,
que está definida por:

$$|| A  ||_F^2 = \sum_{i,j} a_{i,j}^2$$ 
y es una medida de qué tan cerca están las dos matrices $A$ y $B$, componente
a componente. 


### Discusión: aproximación de rango 1.

Empecemos resolviendo el problema más simple, que es

$$\min_{\sigma,u,v} || X - \sigma uv^t ||_F^2$$

donde $\sigma$ es un escalar, $u$ es un vector columna de tamaño $n$ y $v$ es un
vector columna de tamaño $p$. Suponemos que los vectores $u$ y $v$ tienen norma uno.

El objetivo que queremos minimizar es
$$\sum_{i,j} (X_{i,j} - \sigma u_iv_j)^2$$

Derivando con respecto a $u_i$ y $v_j$, e igualando a cero, obtenemos (la sigma
podemos quitarla en la derivada, pues multiplica todo el lado derecho):

$$\frac{\partial}{\partial u_i} = -2\sigma\sum_{j} (X_{i,j} - \sigma u_iv_j)v_j = 0$$
$$\frac{\partial}{\partial v_j} = -2\sigma\sum_{i} (X_{i,j} - \sigma u_iv_j)u_i = 0$$
Que simplificando (y usando que la norma de $u$ y $v$ es igual a 1: $\sum_iu_i^2 = \sum_j v_j^2=1$) quedan:
$$\sum_j X_{i,j}v_j =  \sigma u_i,$$
$$\sum_i X_{i,j}u_i =\sigma  v_j,$$
O en forma matricial 
\begin{equation}
Xv = \sigma u
(\#eq:valor-propio-derecho)
\end{equation}


\begin{equation}
u^t X= \sigma v^t.
(\#eq:valor-propio-izquierdo)
\end{equation}

Podemos resolver este par de ecuaciones para encontrar la solución al problema
de optimización de arriba. Este problema tiene varias soluciones (con distintas $\sigma$), 
pero veremos cómo podemos escoger la que de mejor la aproximación (adelanto:
escoger las solución con $\sigma^2$ más grande). 

\BeginKnitrBlock{comentario}<div class="comentario">A un par de vectores $(u,v)$ que cumplen esta propiedad les llamamos **vector propio
izquierdo** ($u$) y **vector propio derecho** ($v$), con **valor singular** asociado $\sigma$.
Por convención, tomamos $\sigma \geq 0$ (si no, podemos multiplicar a $u$ por menos, por ejemplo).</div>\EndKnitrBlock{comentario}

Y tenemos un resultado importante que nos será útil, y que explica el nombre de estos 
vectores:

\BeginKnitrBlock{comentario}<div class="comentario">Si $(u,v)$ son vectores propios de $X$ asociados a $\sigma$, entonces

- $v$ es un vector propio de la matriz cuadrada $X^tX$ ($p\times p$) con valor propio $\sigma^2$.

- $u$ es un vector propio de la matrix cuadrada $XX^t$ ($n\times n$) con valor propio $\sigma^2$.</div>\EndKnitrBlock{comentario}



**Observaciones**:

- La demostración es fácil pues aplicando $X^t$ a ambos lados de \@ref(eq:valor-propio-derecho), obtenemos
$X^t X v= \sigma X^t u$, que implica $(X^t X) v= \sigma (u^tX)^t = \sigma^2 v$. Podemos
hacer lo mismo para \@ref(eq:valor-propio-izquierdo).

- Nótese que $X^tX$ es una matriz simétrica. Por el teorema espectral, existe una base
ortogonal de vectores propios (usual) $v_1, v_2, \ldots, v_p$ con valores propios
reales. Adicionalmente, como $X^tX$ es positivo-definida, 
entonces todos estos vectores propios tienen valor propio no negativos.




#### Ejemplo {--}
Verifiquemos en el ejemplo del gasto en rubros. Si comparamos $Xv$ con $u$, vemos
que son colineales (es decir, $Xv=\sigma u$):

```r
# qplot(Xv, u), si Xv=sigma*u entonces Xv y u deben ser proporcionales
qplot(as.matrix(X_arr) %*% v_años, u_rubros) + geom_smooth(method='lm')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-29-1.png" width="672" />

Y también

```r
# qplot(u^tX, v^t), si u^tXv=sigma*v entonces Xv y u deben ser proporcionales
qplot(t(as.matrix(X_arr)) %*% u_rubros, (v_años) ) + geom_smooth(method='lm')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-30-1.png" width="672" />


Ahora normalizamos $u$ y $v$ para encontrar $\sigma$:

```r
u_rubros_norm <- normalizar(u_rubros)
v_años_norm <- normalizar(v_años)
(as.matrix(X_arr) %*% v_años_norm)/u_rubros_norm
```

```
##                         [,1]
## Food and Tobacco    123.4858
## Household Operation 123.4855
## Medical and Health  123.4864
## Personal Care       123.4891
## Private Education   123.4799
```

Y efectivamente vemos que $(u,v)$ (normalizados) forman satisfacen las ecuaciones mostradas arriba,
con $\sigma$ igual a:

```r
first((as.matrix(X_arr) %*% v_años_norm)/u_rubros_norm)
```

```
## [1] 123.4858
```

---

Si hay varias soluciones, ¿cuál $\sigma$ escogemos?

Supongamos que encontramos  dos vectores propios $(u,v)$ (izquierdo y derecho)  con 
valor propio asociado $\sigma$.
Podemos evaluar la calidad de la aproximación usando
la igualdad
$$\||A||_F^2 = traza (AA^t)$$
que es fácil de demostrar, pues la componente $(i,i)$ de $AA^t$  está dada por
el producto punto del renglon $i$ de A por el renglón $i$ de $A$, que es $\sum_{i,j}a_{i,j}^2.$

Entonces tenemos que

$$||X-\sigma uv^t||_F^2 = \mathrm{Tr} ((X-\sigma uv^t)(X-\sigma uv^t)^t)$$

que es igual a 

$$ \mathrm{Tr} (XX^t) - 2\sigma \mathrm{Tr} ( X(vu^t)) + \sigma^2\mathrm{Tr}(uv^tvu^t)$$ 

Como $u$ y $v$ tienen norma 1, tenemos que $v^tv=1$, y 
$\textrm{Tr(uu^t)} = \sum_i u_i^2 = 1$.
Adicionalmente, usando el hecho de que $Xv=\sigma u$ obtenemos

$$ ||X-\sigma uv^t||_F^2 = \mathrm{Tr} (XX^t) - \sigma^2$$
que es una igualdad interesante: quiere decir que **la mejor aproximación 
se encuentra encontrando el par de valores propios tal que el valor propio
asociado $\sigma$ tiene el valor $\sigma^2$ más grande posible.** La cantidad
a la cantidad $\mathrm{Tr} (XX^t)$ está dada por
$$\mathrm{Tr} (XX^t) = ||X||_F^2 = \sum_{i,j} X_{i,j}^2,$$
que es una medida del "tamaño" de la matriz $X$.



### Discusión: aproximaciones de rango más alto

Vamos a repetir el análisis para dimensión 2, repitiendo el proceso que hicimos arriba.
Denotamos como $u_1$ y $v_1$ los vectores $u$ y $v$ que encontramos en el paso anterior.
Ahora buscamos minimizar

$$\min_{u_2,v_2} || X - \sigma_1 u_1 v_1^t - \sigma_2 u_2 v_2^{t} ||_F^2$$
Repetimos el argumento de arriba y derivando respecto a las componentes de $u_2,v_2$,
y usando el hecho de que $(u_1, v_1)$ son vectores propios derecho e izquierdo asociados
a $\sigma_1$, obtenemos:

- $v_2$ es ortogonal a $v_1$.
- $u_2$ es ortogonal a $u_1$.
- $(u_2, v_2)$ tienen que ser vectores propios derecho e izquierdo asociados a $\sigma_2\geq 0$.


Usando el hecho de que $v_1$ y $v_2$ son ortogonales, podemos
podemos demostrar igual que arriba que

$$|| X - \sigma_1 u_1 v_1^t - \sigma_2 u_2 v_2^{t} ||_F^2 = \textrm{Tr} (XX^t) - (\sigma_1^2 + \sigma_2^2)$$
De modo que obtenemos la mejor aproximación escogiendo los dos valores de $\sigma_1^2$ y $\sigma_2^2$
más grandes para los que hay solución de \@ref(eq:valor-propio-derecho) y \@ref(eq:valor-propio-izquierdo) y

**Observaciones**:

- Aunque aquí usamos un argumento incremental o *greedy* 
(comenzando con la mejor aproximación de rango 1),
es posible demostrar que la mejor aproximación de rango 2 se puede construir de este modo. Ver
por ejemplo [estas notas](https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf).
- En el caso de dimensión 2, vemos que **la solución es incremental**: $\sigma_1, u_1, v_1$ son los
mismos que para la solución de dimensión 1. En dimensión 2, tenemos que buscar el siguiente
valor singular más grande después de $\sigma_1$, de forma que tenemos $\sigma_1^2 \geq \sigma_2^2$.
La solución entonces es agregar $\sigma_2 u_2 v_2^t$, donde $(u_2,v_2)$ es el par de vectores
propios izquierdo y derecho.

Ahora podemos enunciar nuestro teorema:

\BeginKnitrBlock{comentario}<div class="comentario">**Aproximación de matrices mediante valores singulares**
  
Sea $X$ una matriz $n\times p$, y supongamos que $p\leq n$. Entonces, para cada $k \leq p$, 
  la mejor aproximación de rango $k$ a la  matriz $X$ se puede escribir como
una suma $X_k$ de $k$ matrices de rango 1:
$$X_k =  \sigma_1 u_1v_1^t + \sigma_2 u_2v_2^t + \ldots \sigma_k u_kv_k^t,$$
donde 

- La calidad de la aproximación está dada por 
$$||X-X_k||^2_F = ||X||^2_F - 
  (\sigma_1^2+ \sigma_2^2 + \cdots + \sigma_k^2),$$ de forma que cada aproximación 
es sucesivamente mejor.
- $\sigma_1^2 \geq \sigma_2^2 \geq \cdots  \geq \sigma_k^2\geq 0$
- Los vectores $(u_i,v_i)$ son un par de vectores propios izquierdo y derechos para $X$ con valor singular $\sigma_i$.
- $v_1,\ldots, v_k$ son vectores ortogonales de norma 1
- $u_1,\ldots, u_k$ son vectores ortogonales de norma 1
</div>\EndKnitrBlock{comentario}

**Observaciones**:

1. Normalmente no optimizamos como hicimos en el ejemplo de la matriz de gastos
para encontrar las aproximación de rango bajo,
sino que se usan algoritmos para encontrar vectores propios de $X^tX$ (que son las $v$'s),
o más generalmente algoritmos basados en álgebra lineal 
que intentan encontrar directamente los pares de vectores (u_i, v_i), y otros algoritmos
numéricos (por ejemplo, basados en iteraciones).


2. Un resultado interesante (que faltaría por demostrar) es que si tomamos la aproximación de 
rango $p$ (cuando $p\leq n$), obtenemos que
$$X= \sigma_1 u_1v_1^t + \sigma_2 u_2v_2^t + \ldots \sigma_p u_pv_p^t$$
es decir, la aproximación es exacta. Esto es un fraseo del 
**teorema de descomposición en valores singulares**, que normalmente se expresa
de otra forma (ver más adelante).

### Ejemplo {-}
Consideremos el ejemplo de los gastos. Podemos usar la función *svd* de R

```r
svd_gasto <- svd(X_arr)
```

El objeto de salida contiene los valores singulares (en *d*). Nótese que ya habíamos
calculado por fuerza bruta el primer valor singular:


```r
sigma <- svd_gasto$d
sigma
```

```
## [1] 123.4857584   4.5673718   0.3762533
```

Los vectores $v_1,v_2,v_3$ (pesos de las variables) en nuestras tres nuevas dimensiones, 
que son las columnas de


```r
v <- svd_gasto$v
rownames(v) <- colnames(X_arr)
v
```

```
##            [,1]       [,2]        [,3]
## 1940 -0.2007388 -0.3220495 -0.92519623
## 1950 -0.5423269 -0.7499672  0.37872247
## 1960 -0.8158342  0.5777831 -0.02410854
```
y los vectores $u_1,u_2,u_3$, que son los scores de los rubros en cada dimensión


```r
dim(svd_gasto$u)
```

```
## [1] 5 3
```

```r
u <- (svd_gasto$u)
rownames(u) <- rownames(X_arr)
u
```

```
##                            [,1]       [,2]       [,3]
## Food and Tobacco    -0.87130286 -0.3713244 -0.1597823
## Household Operation -0.44966139  0.3422116  0.4108311
## Medical and Health  -0.18778444  0.8259030 -0.2584369
## Personal Care       -0.04812680  0.2074885 -0.4372590
## Private Education   -0.03250802  0.1408623  0.7400691
```

Podemos considerar ahora la segunda dimensión que encontramos. 

- En los scores: $u_2$ tiene valores altos en el rubro 3 (salud), y valores negativos en rubro 1. Es un patrón
de gasto más alto en todo menos en comida (que es el rubro 1), especialmente en salud.

- Ahora vemos $v_2$:  tiene un valor alto en el año 60 (3a entrada), y valores más negativos para
los dos primeros años (40 y 50)

- Así que decimos que en los 60, el ingreso se desplazó hacia salud (y otros rubros en general),
reduciéndose el de comida.

 Si multiplicamos podemos ver la contribución de esta matriz de rango 1 (en billones (US) de dólares):

```r
d <- svd_gasto$d
(d[2]*tcrossprod(svd_gasto$u[,2], svd_gasto$v[,2])) %>% round(1)
```

```
##      [,1] [,2] [,3]
## [1,]  0.5  1.3 -1.0
## [2,] -0.5 -1.2  0.9
## [3,] -1.2 -2.8  2.2
## [4,] -0.3 -0.7  0.5
## [5,] -0.2 -0.5  0.4
```

Este es un efecto relativamente chico (comparado con el patrón estable de la primera
dimensión), pero ilumina todavía un aspecto adicional de esta tablita.

La norma de la diferencia entre la matriz $X$ y la aproximación de rango 2 podemos calcularla
de dos maneras:


```r
sum(X_arr^2) - sum(d[1:2]^2)
```

```
## [1] 0.1415665
```
O calculando la aproximación y la diferencia directamente. Podemos hacerlo de la siguiente forma


```r
X_arr_2 <- d[1]*tcrossprod(u[,1], v[,1]) + d[2]*tcrossprod(u[,2], v[,2])
sum((X_arr - X_arr_2)^2)
```

```
## [1] 0.1415665
```
Pero podemos calcular la aproximación $X_2$ en forma matricial, haciendo

```r
X_arr_2 <- u[,1:2] %*% diag(d[1:2]) %*% t(v[,1:2])
sum((X_arr - X_arr_2)^2)
```

```
## [1] 0.1415665
```

## Descomposición en valores singulares (SVD o DVS)

Aunque ya hemos enunciado los resultados, podemos enunciar el teorema de descomposición
en valores singulares en términos matriciales.

Supongamos entonces que tenemos una aproximación de rango $k$

$$X_k = \sigma_1 u_1v_1^t + \sigma_2 u_2v_2^t + \ldots \sigma_k u_kv_k^t$$

Se puede ver que esta aproximación se escribe como (considera todos los vectores
como vectores columna)

$$ X_k = (u_1,u_2, \ldots, u_k) \left(     
{\begin{array}{ccccc}
\sigma_1 & 0 & \cdots & \cdots & 0 \\
0 & \sigma_2 & 0 &\cdots & 0 \\
\vdots & & & \vdots\\
0 & 0 & 0 & \cdots & \sigma_k  \\
\end{array} }
\right)
\left (
\begin{array}{c}
v_1^t \\
v_2^t \\
\vdots \\
v_k^t
\end{array}
\right)
$$

o más simplemente, como

$$X_k = U_k \Sigma_k V_k^t$$
donde $U_k$ ($n\times k$) contiene los vectores $u_i$ en sus columnas, 
$V_k$ ($k\times p$) contiene los vectores $v_j$ en sus columnas, y
la matriz $\Sigma_k$ es la matriz diagonal con los primeros $\sigma_1\geq \sigma_2\geq\cdots \sigma_k$
valores singulares.

Ver el ejemplo anterior para ver cómo los cálculos son iguales.

\BeginKnitrBlock{comentario}<div class="comentario">**Descomposición en valores singulares**
  
Sea $X$ una matriz de $n\times p$ con $p\leq n$. Entonces existe una factorización
$$X=U\Sigma V^t,$$

- $\Sigma$ es una matriz diagonal con valores no-negativos (valores singulares).
Los valores singulares de $\Sigma$ estan ordenados en orden decreciente.
    
- Las columnas de U y V son vectores ortogonales unitarios. La i-ésima columna
$u_i$ de $V$ y la í-esima columna $v_i$ de $V$ son pares de vectores propios $(u_i, v_i)$ izquierdo y derecho de $X$  con valor singular $\sigma_i = \Sigma_{i,i}$
</div>\EndKnitrBlock{comentario}

- Una vez que tenemos esta descomposición, podemos extraer la aproximación que
nos sea útil: una aproximación  $X_k$ de orden $k$ se escribe como
$$X_k = U_k\Sigma_k V_k^t$$
donde $U_k$ contiene las primeras $k$ columnas de $U$, $V_k$ las primeras $k$
columnas de $V$, y $\Sigma_k$ es la submatriz cuadrada $k\times k$ de los primeros
$k$ renglones y columnas de $\Sigma$ :


```r
knitr::include_graphics("imagenes/svd.png")
```

<img src="imagenes/svd.png" width="500" />


- Frecuenta el teorema de aproximación óptima (teorema de Ekhart-Young) 
se deriva de la descomposición en valores singulares, que se demuestra antes
usando técnicas de álgebra lineal.



## Interpretación geométrica

La descomposición en valores singulares también se puede interpretar geométricamente.
Para ver cómo funciona, primero observamos que:

\BeginKnitrBlock{comentario}<div class="comentario">**Proyecciones**

Los vectores $v_1,v_2, \ldots, v_p$ están en el espacio de variables o columnas
(son de dimensión $p$). La componente de la proyección 
(ver [proyección de vectores](https://en.wikipedia.org/wiki/Vector_projection) )
de la matriz de datos sobre una de estas dimensiones está dada por
$$Xv_j,$$
que son iguales a los scores de los casos escalados por $\sigma$:
$$\sigma_j u_j$$. 

Las proyecciones $d_j = \sigma_j u_j$
  son las variables que normalmente se usan para hacer análisis posterior,
aunque cuando la escala de las proyecciones no es importante,
también se pueden usar simplemente las $u_j$.</div>\EndKnitrBlock{comentario}

Por ejemplo, la projeccion del rengón $x_i$ de la matriz $X$ es
$(x_i^tv_j) v_j$ (nótese que $x_i^tv_j$ es un escalar, la componente de la proyección).

Consideremos unos datos simulados


```r
set.seed(3221)
x_1 <- rnorm(200,2, 1)
x_2 <- rnorm(200,0,1) + x_1
datos <- data_frame(x_1, x_2)
ggplot(datos, aes(x=x_1, y=x_2)) + geom_point() +
  geom_vline(xintercept = 0, colour='red') +
 geom_hline(yintercept = 0, colour='red')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-45-1.png" width="672" />

Hacemos descomposición en valores singulares y graficamos 

```r
svd_x <- svd(datos)
v <- svd_x$v %>% t %>% as.data.frame() 
u <- svd_x$u %>% data.frame
colnames(v) <- c('x_1','x_2')
colnames(u) <- c('x_1','x_2')
d <- svd_x$d
#Nota: podemos mover signos para hacer las gráficas y la interpetación
# más simples
v[,1] <- - v[,1]
u[,1] <- - u[,1]
v
```

```
##         x_1        x_2
## 1 0.6726219 -0.7399864
## 2 0.7399864  0.6726219
```

Graficamos ahora los dos vectores $v_1$ y $v_2$, escalándolos
para ver mejor cómo quedan en relación a los datos (esto no es necesario
hacerlo):


```r
ggplot(datos) + geom_point(aes(x=x_1, y=x_2)) +
  geom_vline(xintercept = 0, colour='red') +
 geom_hline(yintercept = 0, colour='red') + 
  geom_segment(data = v, aes(xend= 4*x_1, yend=4*x_2, x=0, y=0), col='red', size=1.1,
               arrow = arrow(length = unit(0.3,"cm")))  +
  coord_equal()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-47-1.png" width="672" />


- El primer vector es el "que pasa más cercano a los puntos", en el sentido de que la distancia
entre los datos proyectados al vector y los datos es lo más chica posible (mejor aproximación). La
proyección de los datos sobre $v$ es igual a $Xv_1=\sigma_1 u_1$, es decir, está dada por 
$\sigma u_1$

- Las proyecciones de los datos sobre el segundo vector $v_2$ están dadas igualmente por 
$\sigma_2 u_2$. Sumamos esta proyección a la de la primera dimensión para obtener una mejor
aproximación a los datos (en este caso, exacta).

Por ejemplo, seleccionemos el primer punto y obtengamos sus proyecciones:


```r
proy_1 <- (d[1])*u[1,1]*v[,1] #v_1 por el score en la dimensión 1 u[1,1]
proy_2 <- (d[2])*u[1,2]*v[,2] #v_2 por el score en la dimensión 1 u[1,1]
proy_2 + proy_1
```

```
## [1] 3.030313 1.883698
```

```r
datos[1,]
```

```
## # A tibble: 1 x 2
##        x_1      x_2
##      <dbl>    <dbl>
## 1 3.030313 1.883698
```

Podemos graficar la aproximación sucesiva:


```r
datos$selec <- c('seleccionado', rep('no_seleccionado', nrow(datos)-1))
ggplot(datos) + geom_point(aes(x=x_1, y=x_2, colour=selec, size=selec)) +
  geom_vline(xintercept = 0, colour='red') +
 geom_hline(yintercept = 0, colour='red') + 
  geom_segment(aes(xend= proy_1[1], yend=proy_1[2], x=0, y=0), col='red', size=1.1,
               arrow = arrow(length = unit(0.3,"cm")))  +
  geom_segment(aes(xend= proy_2[1] + proy_1[1], yend=proy_2[2] + proy_1[2], 
                     x=proy_1[1], y=proy_1[2]), 
                col='red', size=1.1,
                arrow = arrow(length = unit(0.2,"cm")))  +
  coord_equal()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-49-1.png" width="672" />

\BeginKnitrBlock{comentario}<div class="comentario">- Las aproximaciones de la descomposión en valores singulares mediante matrices de rango 1 puede
entenderse como la búsqueda sucesiva de subespacios de dimensión baja, donde al proyectar los datos perdemos
poca información. 
- Las proyecciones sucesivas se hacen sobre vectores ortogonales, y en este sentido
la DVS separa la información en partes que no tienen contenido común (desde el punto
 de vista lineal).</div>\EndKnitrBlock{comentario}

Finalmente, muchas veces graficamos las proyecciones en el nuevo espacio creado
por las dimensiones de la DVS (nótese la escala distinta de
los ejes).


```r
proyecciones <- data_frame(dim_1 = d[1]*u[,1], dim_2 = d[2]*u[,2],
                          selec = datos$selec) 
ggplot(proyecciones, aes(x = dim_1, y = dim_2, size=selec, colour=selec)) + 
  geom_point() 
```

```
## Warning: Using size for a discrete variable is not advised.
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-51-1.png" width="672" />



## SVD para películas de netflix

Vamos a intentar encontrar dimensiones latentes para los datos del concurso
de predicción de Netflix (una de las componentes de las soluciones ganadoras
fue descomposición en valores singulares).


```r
#no correr en notas - son unas 50 millones de evaluaciones
# puedes bajar los datos y reproducir desde datos originales bajando el archivo
# https://s3.amazonaws.com/netflix-am2017/muestra_calificaciones_1.csv
if(FALSE){
  evals <- read_csv('datos/netflix/muestra_calificaciones_1.csv')
  evals
}
```


```r
peliculas_nombres <- read_csv('datos/netflix/peliculas_1.csv')
```

```
## Parsed with column specification:
## cols(
##   pelicula_id = col_integer(),
##   year = col_integer(),
##   name = col_character()
## )
```

```r
peliculas_nombres
```

```
## # A tibble: 646 x 3
##    pelicula_id  year                                name
##          <int> <int>                               <chr>
##  1           1  2003              Something's Gotta Give
##  2           2  1992                      Reservoir Dogs
##  3           3  2003                    X2: X-Men United
##  4           4  2004                        Taking Lives
##  5           5  1959                  North by Northwest
##  6           6  2004 Harold and Kumar Go to White Castle
##  7           7  2001               Bridget Jones's Diary
##  8           8  2000                       High Fidelity
##  9           9  2000                      Pay It Forward
## 10          10  1999                               Dogma
## # ... with 636 more rows
```

Hay muchas peliculas que no son evaluadas por ningún usuario. Aquí tenemos que decidir
cómo tratar estos datos: si los rellenamos con 0, la implicación es que un usuario
tiene bajo interés en una película que no ha visto. Hay otras opciones (y quizá un
método que trate apropiadamente los datos faltantes es mejor).


```r
#no correr en notas
library(Matrix)
library(methods)
library(irlba)
if(FALSE){
  evals <- evals %>% group_by(usuario_id) %>% mutate(calif_centrada = calif - mean(calif))
  #Usamos matriz rala - de otra manera la matriz es demasiado grande
  evals_mat <- sparseMatrix(i = evals$usuario_id, j=evals$pelicula_id, x = evals$calif)
  svd_parcial <- irlba(evals_mat, 4)
  saveRDS(svd_parcial, file ='cache_obj/svd_netflix.rds')
}
```


```r
svd_parcial <- readRDS('cache_obj/svd_netflix.rds')
svd_parcial$d
```

```
## [1] 16853.865  5346.353  4170.122  3970.022
```


```r
#no correr en notas
if(FALSE){
V_peliculas <- data_frame(v_1 = svd_parcial$v[,1], v_2 = svd_parcial$v[,2],
                     v_3 = svd_parcial$v[,3], v_4 = svd_parcial$v[,4],
                     pelicula_id=1:ncol(evals_mat)) %>% 
                left_join(peliculas_nombres)
U_usuarios <- data_frame(u_1 = svd_parcial$u[,1], v_2=svd_parcial$u[,2],
                         u_3 = svd_parcial$u[,3], u_4 = svd_parcial$u[,4],
                         usuario_id = 1:nrow(evals_mat))
saveRDS(V_peliculas, file = 'cache_obj/v_peliculas.rds')
saveRDS(U_usuarios, file = 'cache_obj/u_usuarios.rds')
}
```


```r
V_peliculas <- readRDS('cache_obj/v_peliculas.rds')
U_usuarios <- readRDS('cache_obj/u_usuarios.rds')
```

Veamos primero las componentes 2, 3 y 4. 



```r
qplot(svd_parcial$u[,2])
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-58-1.png" width="672" />

```r
qplot(svd_parcial$v[,2])#
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-58-2.png" width="672" />



```r
library(ggrepel)
pel_graf <- V_peliculas %>% mutate(dist_0 = sqrt(v_2^2 + v_3^2)) 
muestra <- pel_graf %>% mutate(etiqueta = ifelse(dist_0 > 0.08, name, ''))
ggplot(muestra, aes(x=v_2, y=v_3, label=etiqueta)) + geom_point(alpha=0.2) + 
  geom_text_repel(size=2.5) + xlab('Mainstream vs Independiente') + ylab('Violenta/Acción vs Drama')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-59-1.png" width="672" />


```r
pel_graf <- V_peliculas %>% mutate(dist_0 = sqrt(v_3^2 + v_4^2)) 
muestra <- pel_graf %>% mutate(etiqueta = ifelse(dist_0 > 0.08, name, ''))
ggplot(muestra, aes(x=v_3, y=v_4, label=etiqueta)) + geom_point(alpha=0.2) + 
  geom_text_repel(size=2.5)  + xlab('Violenta/Acción vs Drama') + ylab('Fantasía/Ciencia Ficción')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-60-1.png" width="672" />



Dejamos la primer componente porque es más bien consecuencia de cómo construimos
la matriz que buscamos descomponer:


```r
qplot(svd_parcial$u[,1])
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-61-1.png" width="672" />

```r
qplot(svd_parcial$v[,1])
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-61-2.png" width="672" />
Esta componente está asociada con el número de evaluaciones que tiene cada
usuario y que tiene cada persona


```r
if(FALSE){
  evals_num_u <- evals %>% group_by(usuario_id) %>% summarise(num_evals = n())
  saveRDS(evals_num_u, 'cache_obj/evals_num_u.rds')
}
```


```r
evals_num_u <- readRDS('cache_obj/evals_num_u.rds')
qplot(evals_num_u$num_evals, svd_parcial$u[,1])
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-63-1.png" width="672" />


```r
if(FALSE){
  evals_num_p <- evals %>% group_by(pelicula_id) %>% summarise(num_evals = n(), calif_prom=mean(calif))
  saveRDS(evals_num_p, 'cache_obj/evals_num_p.rds')
}
```


```r
evals_num_p <- readRDS('cache_obj/evals_num_p.rds')
qplot(evals_num_p$num_evals, svd_parcial$v[,1])
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-65-1.png" width="672" />

Esta dimensión aparece pues la primera aproximación de rango 1 intenta replicar
los valores "bajos" de pocas evaluaciones tanto en usuarios como en películas.
En realidad es una distorsión producida por cómo hemos tratado los datos ("imputando"
cero cuando no existe una evaluación).

### Calidad de representación de SVD.

Podemos hacer varios cálculos para entender qué tan buena es nuestra aproximación de
rango bajo $X_k$. Por ejemplo, podríamos calcular las diferencias de $X-X_k$ y presentarlas
de distinta forma.

### Ejemplo{-}
En el ejemplo de rubros de gasto, podríamos mostrar las diferencias en billones (us)
de dólares, donde vemos que la aproximación es bastante buena


```r
qplot(as.numeric(X_arr-X_arr_2))
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-66-1.png" width="672" />

Que podríamos resumir, por ejemplo, con la media de errores absolutos:


```r
mean(abs(as.numeric(X_arr-X_arr_2)))
```

```
## [1] 0.06683576
```

Otra opción es usar la norma Frobenius, calculando para la apoximación de rango 1



```r
1 - (sum(X_arr^2) - sum(svd_gasto$d[1]^2))/sum(X_arr^2)
```

```
## [1] 0.9986246
```

Lo que indica que capturamos 99.8\% de la información, y para la de rango 2:
d

```r
1-(sum(X_arr^2) - sum(svd_gasto$d[1:2]^2))/sum(X_arr^2)
```

```
## [1] 0.9999907
```

Lo que indica que estos datos (en 3 variables), podemos entenderlos mediante
un análisis de dos dimensiones

---

\BeginKnitrBlock{comentario}<div class="comentario">Podemos medir la calidad de la representación de $X$ ($n\times p$ con $p < n$) 
de una aproximación $X_k$ de SVD mediante
$$1-\frac{||X-X_k||_F^2}{||X||_F^2}  = \frac{\sigma_1^2 + \sigma_2^2 + \cdots \sigma_k^2}{\sigma_1^2 + \sigma_2^2 + \cdots \sigma_p^2},$$
que es un valor entre 0 y 1. Cuanto más cercana a 1 está, mejor es la representación.</div>\EndKnitrBlock{comentario}

**Observaciones**: Dependiendo de nuestro objetivo, nos interesa alcanzar distintos
niveles de calidad de representación. Por ejemplo, algunas reglas de dedo:

- Si queremos usar los datos para un proceso posterior, o dar una descripción 
casi completa de los datos, quizá buscamos calidad $>0.9$ o mayor.

- Si nos interesa extraer los patrones más importantes, podemos considerar valores de calidad
mucho más chicos, entendiendo que hay una buena parte de la información que no se explica
por nuestra aproximación.


#### Ejemplo {-}
Para el problema de Netflix podemos calcular la calidad de representación dada por
las primeras 4 dimensiones


```r
#norma_X  <- sum(evals$calif^2)
norma_X <- 738958222
sum(svd_parcial$d^2)/norma_X
```

```
## [1] 0.4679388
```

Lo que indica que todavía hay mucha información por explorar en los datos de netflix. La
contribución a este porcentaje de cada dimensión


```r
(svd_parcial$d[1]^2)/norma_X
```

```
## [1] 0.3843962
```

```r
(svd_parcial$d[2]^2)/norma_X
```

```
## [1] 0.0386808
```

```r
(svd_parcial$d[3]^2)/norma_X
```

```
## [1] 0.02353302
```

```r
(svd_parcial$d[3]^2)/norma_X
```

```
## [1] 0.02353302
```




## Componentes principales

Componentes principales es la descomposición en valores singulares aplicada
a una matriz de datos centrada por columna. Esta operación convierte el problema
de aproximación de matrices de rango bajo en uno de aproximaciones que buscan
explicar la mayoría de la *varianza* (incluyendo covarianza) 
de las variables de la matriz de datos $X$. 


Consideremos entonces una matriz de datos $X$ de tamaño $n\times p$. Definimos
la **matrix centrada** por columna $\tilde{X}$ , que se calcula como
$$\tilde{X}_{i,j} = X_{i,j} - \mu_j$$
  donde $\mu_j = \frac{1}{n} \sum_j X_{i,j}$.


- La diferencia en construcción entre svd y svd con columnas centradas (componentes principales)
es que en svd las proyecciones se hacen pasando por el origen, pero en componentes principales se hacen a partir del centroide de los datos

### Ejemplo {-}

Veamos primero el último ejemplo simulado que hicimos anterioremnte. Primero centramos
los datos por columna:
  

```r
datos_c <- scale(datos %>% select(-selec), scale = FALSE) %>% as.data.frame
ggplot(datos_c, aes(x=x_1, y=x_2)) + geom_point() +
  geom_vline(xintercept = 0, colour='red') +
  geom_hline(yintercept = 0, colour='red')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-73-1.png" width="672" />

Y ahora calculamos la descomposición en valores singulares


```r
svd_x <- svd(datos_c)
v <- svd_x$v %>% t %>% as.data.frame() 
u <- svd_x$u %>% data.frame
colnames(v) <- c('x_1','x_2')
colnames(u) <- c('x_1','x_2')
d <- svd_x$d
v
```

```
##        x_1       x_2
## 1 0.507328  0.861753
## 2 0.861753 -0.507328
```

Notemos que los resultados son similares, pero no son los mismos.

Graficamos ahora los dos vectores $v_1$ y $v_2$, que en este contexto
se llaman *direcciones principales*
  

```r
ggplot(datos_c) + geom_point(aes(x=x_1, y=x_2)) +
  geom_vline(xintercept = 0, colour='red') +
  geom_hline(yintercept = 0, colour='red') + 
  geom_segment(data = v, aes(xend= 5*x_1, yend=5*x_2, x=0, y=0), col='red', size=1.1,
               arrow = arrow(length = unit(0.3,"cm")))  +
  coord_equal()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-75-1.png" width="672" />

Las componentes de las proyecciones de los datos sobre las direcciones principales dan las
**componentes principales** (nótese que multiplicamos por los valores singulares):
  

```r
head(svd_x$u %*% diag(svd_x$d))
```

```
##            [,1]       [,2]
## [1,]  0.3230641  0.9257829
## [2,]  0.4070429  1.5360770
## [3,] -1.2788977 -0.2762829
## [4,]  0.8910247  0.4071926
## [5,] -4.4466993 -0.5743111
## [6,] -1.2267878  0.3470759
```

Que podemos graficar


```r
comps <- svd_x$u %*% diag(svd_x$d) %>% data.frame
ggplot(comps, aes(x=X1, y=X2)) + geom_point()+
  geom_vline(xintercept = 0, colour='red') +
  geom_hline(yintercept = 0, colour='red')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-77-1.png" width="672" />

Este resultado lo podemos obtener directamente usando la función *princomp*
  

```r
comp_principales <- princomp(datos %>% select(-selec))
scores <- comp_principales$scores
head(scores)
```

```
##          Comp.1     Comp.2
## [1,]  0.3230641 -0.9257829
## [2,]  0.4070429 -1.5360770
## [3,] -1.2788977  0.2762829
## [4,]  0.8910247 -0.4071926
## [5,] -4.4466993  0.5743111
## [6,] -1.2267878 -0.3470759
```

Y verificamos que los resultados son los mismos:

```r
qplot(scores[,1], comps[,1])
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-79-1.png" width="384" />

```r
qplot(scores[,2], -comps[,2])
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-79-2.png" width="384" />

---
  
### Varianza en componentes principales.
  
  Cuando centramos por columna, la svd es un tipo de análisis de la matriz
de varianzas y covarianzas de la matriz $X$, dada por

$$C = \frac{1}{n} \tilde{X}^t \tilde{X}$$
  (Nota: asegúrate de que entiendes por qué esta es la matriz de varianzas y covarianzas
   de $X$). 

\BeginKnitrBlock{comentario}<div class="comentario">Nótese que las proyecciones (que se llaman **componentes principales**) 
$\tilde{X}v_j = \sigma_j u_j = d_j$ satisfacen que

1. La media de las proyecciones $d_j$ es igual a cero

Pues $$\sigma_j \sum_k {u_{j,k}} = \sum_k \sum_i (\tilde{X}_{k,i})v_{j,i} =
  \sum_i v_{j,i}\sum_k (\tilde{X}_{k,i}) = 0,$$
  pues las columnas de $\tilde{X}$ tienen media cero.

2- $\sigma_j^2$ es la varianza de la proyección $d_j$, pues
$$Var(d_j) = \sigma_j^2 \sum_k (u_{j,k} - 0)^2 = \sigma_j^2,$$
  y el vector $u_j$ tienen norma 1.

3. La ortogonalidad de los vectores $u_j$ se interpreta ahora en términos
de covarianza:
  $$Cov(d_i,d_j) = \frac{1}{n}\sum_{k=1}^n (d_{i,k}-0)(d_{j,k}-0) =
  \frac{1}{n}\sum_{k=1}^n \sigma_j\sigma_i u_{i,k}u_{j,k} = 0$$</div>\EndKnitrBlock{comentario}
  

Así que 
  
\BeginKnitrBlock{comentario}<div class="comentario">  Buscamos sucesivamente direcciones para proyectar *que tienen
varianza máxima* (ver ejemplo anterior), y que sean no correlacionadas de forma
que no compartan información lineal entre ellas.</div>\EndKnitrBlock{comentario}

Adicionalmente, vimos que podíamos escribir
$$||\tilde{X}||^2_F =  \sum_{j=1}^p \sigma_{j}^2$$
  Y el lado izquierdo es en este caso una suma de varianzas:
  $$\sum_{j=1}^p Var(X_j) =  \sum_{j=1}^p \sigma_{j}^2.$$
  El lado izquierdo se llama *Varianza total* de la matriz $X$. Componentes principales particiona la varianza total de la matriz $X$ en 
componentes .

## ¿Centrar o no centrar por columna?

Típicamente, antes de aplicar SVD hacemos algunos pasos de procesamiento
de las variables. En componentes principales, este paso de procesamiento
es centrar la tabla por columnas. Conviene hacer esto cuando:
  
1. Centramos si las medias de las columnas no tienen información importante o interesante
para nuestros propósitos - es mejor eliminar esta parte de variación desde el principio para no 
lidiar con esta información en 
las dimensiones que obtengamos. En otro caso, quizá es mejor no centrar.

2. Centramos si nos interesa más tener una interpretación en términos de varianzas y covarianzas que hacer una aproximación de los datos originales.

Sin embargo, también es importante notar que muchas veces **los resultados de ambos
análisis son similares** en cuanto a interpretación y en cuanto a usos posteriores
de las dimensiones obtenidas.


Pueden ver análisis detallado en [este artículo](https://www.researchgate.net/publication/255644479_On_Relationships_Between_Uncentred_And_Column-Centred_Principal_Component_Analysis), que
hace comparaciones a lo largo de varios conjuntos de datos.



### Ejemplo: resultados similares{-}

En el ejemplo de gasto en rubros que vimos arriba, los pesos $v_j$ son muy similares:
  

```r
comps_1 <- princomp(USPersonalExpenditure[,c(1,3,5)])
svd_1 <- svd(USPersonalExpenditure[,c(1,3,5)])
comps_1$loadings[,]
```

```
##          Comp.1     Comp.2      Comp.3
## 1940 -0.2099702 -0.2938755  0.93249650
## 1950 -0.5623341 -0.7439168 -0.36106546
## 1960 -0.7998081  0.6001875  0.00905589
```

```r
svd_1$v
```

```
##            [,1]       [,2]        [,3]
## [1,] -0.2007388 -0.3220495 -0.92519623
## [2,] -0.5423269 -0.7499672  0.37872247
## [3,] -0.8158342  0.5777831 -0.02410854
```


```r
comps_1$scores
```

```
##                        Comp.1     Comp.2      Comp.3
## Food and Tobacco    -68.38962 -0.8783065  0.06424607
## Household Operation -16.25334  0.9562769 -0.16502902
## Medical and Health   16.13276  2.2900370  0.07312028
## Personal Care        33.29512 -1.0003211  0.23036177
## Private Education    35.21507 -1.3676863 -0.20269910
```

```r
svd_1$u %*% diag(svd_1$d)
```

```
##             [,1]       [,2]        [,3]
## [1,] -107.593494 -1.6959766 -0.06011861
## [2,]  -55.526778  1.5630078  0.15457655
## [3,]  -23.188704  3.7722060 -0.09723774
## [4,]   -5.942974  0.9476773 -0.16452015
## [5,]   -4.014277  0.6433704  0.27845344
```

Llegaríamos a conclusiones similares si interpretamos cualquiera de los dos análisis
(verifica por ejemplo el ordenamiento de rubros y años en cada dimensión).

## Ejemplos: donde es buena idea centrar  {-}

Por ejemplo, si hacemos componentes principales con los siguientes datos:
  

```r
whisky <- read_csv('./datos/whiskies.csv')
head(whisky)
```

```
## # A tibble: 6 x 17
##   RowID  Distillery  Body Sweetness Smoky Medicinal Tobacco Honey Spicy
##   <chr>       <chr> <int>     <int> <int>     <int>   <int> <int> <int>
## 1    01   Aberfeldy     2         2     2         0       0     2     1
## 2    02    Aberlour     3         3     1         0       0     4     3
## 3    03      AnCnoc     1         3     2         0       0     2     0
## 4    04      Ardbeg     4         1     4         4       0     0     2
## 5    05     Ardmore     2         2     2         0       0     1     1
## 6    06 ArranIsleOf     2         3     1         1       0     1     1
## # ... with 8 more variables: Winey <int>, Nutty <int>, Malty <int>,
## #   Fruity <int>, Floral <int>, Postcode <chr>, Latitude <int>,
## #   Longitude <int>
```


```r
whisky_sabor <- whisky %>% select(Body:Floral)
comp_w <- princomp(whisky_sabor)
```

Veamos los pesos de las primeras cuatro dimensiones

```r
round(comp_w$loadings[,1:4],2)
```

```
##           Comp.1 Comp.2 Comp.3 Comp.4
## Body        0.36  -0.49  -0.03   0.07
## Sweetness  -0.20  -0.05   0.26   0.37
## Smoky       0.48  -0.07  -0.22  -0.09
## Medicinal   0.58   0.16  -0.04  -0.08
## Tobacco     0.09   0.02   0.00   0.03
## Honey      -0.22  -0.42  -0.11  -0.03
## Spicy       0.06  -0.18  -0.70   0.17
## Winey      -0.04  -0.64   0.23   0.23
## Nutty      -0.05  -0.26   0.18  -0.85
## Malty      -0.13  -0.10  -0.11  -0.07
## Fruity     -0.20  -0.12  -0.40  -0.09
## Floral     -0.38   0.13  -0.34  -0.15
```

La primera componente separa whisky afrutado/floral/dulce de los whishies ahumados
con sabor medicinal. La segunda componente separa whiskies con más cuerpo, características
de vino y miel de otros más ligeros. Las siguientes componentes parece oponer
*Spicy* contra *Fruity* y *Floral*, y la tercera principalmente contiene la medición de *Nutty*.

Según vimos arriba, podemos ver que porcentaje de la varianza explica cada componente


```r
summary(comp_w)
```

```
## Importance of components:
##                           Comp.1    Comp.2     Comp.3     Comp.4
## Standard deviation     1.5268531 1.2197972 0.86033607 0.79922719
## Proportion of Variance 0.3011098 0.1921789 0.09560193 0.08250322
## Cumulative Proportion  0.3011098 0.4932887 0.58889059 0.67139381
##                            Comp.5    Comp.6     Comp.7     Comp.8
## Standard deviation     0.74822104 0.6811330 0.62887454 0.59593956
## Proportion of Variance 0.07230864 0.0599231 0.05108089 0.04587064
## Cumulative Proportion  0.74370245 0.8036256 0.85470644 0.90057708
##                            Comp.9    Comp.10    Comp.11     Comp.12
## Standard deviation     0.52041611 0.49757158 0.42174644 0.271073661
## Proportion of Variance 0.03498097 0.03197728 0.02297382 0.009490848
## Cumulative Proportion  0.93555805 0.96753533 0.99050915 1.000000000
```

Y vemos que las primeras dos componentes explican casi el 50\% de la varianza. Las
siguientes componentes aportan relativamente pooca varianza comparada con la primera

Podemos graficar los whiskies en estas dos dimensiones:


```r
library(ggrepel)
scores_w <- comp_w$scores %>% data.frame
scores_w$Distillery <- whisky$Distillery
ggplot(scores_w, aes(x=Comp.1, y= -Comp.2, label=Distillery)) + 
  geom_vline(xintercept=0, colour = 'red') +
  geom_hline(yintercept=0, colour = 'red') +
  geom_point()+
  geom_text_repel(size=2.5, segment.alpha = 0.3, force = 0.1, seed=202) +
  xlab('Fruity/Floral vs. Smoky/Medicional') +
  ylab('Winey/Body and Honey')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-88-1.png" width="672" />


¿Que pasa si usamos svd sin centrar? Vemos que la primera componente simplemente
captura los distintos niveles promedio de las variables.  Esta componente *no es muy
interesante*, pues por las características del whisky es normal que Medicinal o Tabaco
tengo una media baja, comparado con dulzor, Smoky, etc. Adicionalmente,
el vector $u$ asociado a esta dimensión tiene poca variación:


```r
svd_w <- svd(whisky_sabor)
svd_w$v[,1:2]
```

```
##              [,1]        [,2]
##  [1,] -0.39539241 -0.38286900
##  [2,] -0.42475240  0.19108176
##  [3,] -0.28564145 -0.48775482
##  [4,] -0.09556061 -0.57453247
##  [5,] -0.02078706 -0.09187451
##  [6,] -0.24191199  0.20518808
##  [7,] -0.26485793 -0.07103866
##  [8,] -0.19488910  0.01930294
##  [9,] -0.27858538  0.03430020
## [10,] -0.33669488  0.11644937
## [11,] -0.34001725  0.18933847
## [12,] -0.31441595  0.37730436
```

```r
plot(svd_w$v[,1], apply(whisky_sabor, 2, mean))
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-89-1.png" width="672" />

```r
mean(svd_w$u[,1])
```

```
## [1] -0.1063501
```

```r
sd(svd_w$u[,1])
```

```
## [1] 0.01792508
```

*Observación* La primera componente de svd está haciendo el trabajo de ajustar la media.
Como no nos interesa este hecho, podemos mejor centrar desde el principio y trabajar
con las componentes principales. ¿Cómo se ven las siguientes dos dimensiones
del análisis no centrado?

## Ejemplo: donde no centrar funciona bien  {-}

Considera el ejemplo de la tarea con la tabla de gastos en distintas categorías
de alimentos según el decil de ingreso del hogar. ¿Por qué en este ejemplo centrar
por columna no es tan buena idea? Si hacemos el centrado, quitamos
información importante de la tabla, que es que los distintos deciles
tienen distintos niveles de gasto. 

Veamos como lucen los dos análisis. Para componentes principales:


```r
deciles <- read_csv('./datos/enigh_deciles.csv') %>% as.data.frame
```

```
## Warning: Missing column names filled in: 'X1' [1]
```

```r
deciles %>% arrange(desc(d1))
```

```
##                                  X1      d1      d2      d3      d4
## 1                          CEREALES 1330728 1869247 2254304 2331371
## 2                            CARNES 1072718 1754012 2131706 2514365
## 3  VERDURAS, LEGUMBRES, LEGUMINOSAS  973984 1279986 1478179 1590063
## 4             LECHE Y SUS DERIVADOS  585910  895216 1242102 1395102
## 5          OTROS ALIMENTOS DIVERSOS  290038  448629  689605  781629
## 6                             HUEVO  255321  360471  421613  442603
## 7                            FRUTAS  192462  283549  337608  468187
## 8                   AZUCAR Y MIELES  167042  212941  200200  191048
## 9                  ACEITES Y GRASAS  135823  190052  179945  183546
## 10              PESCADOS Y MARISCOS  110398  187546  213830  236001
## 11                       TUBERCULOS  107231  158078  190705  201664
## 12             CAFE, TE Y CHOCOLATE   71945  120338  108609   97139
## 13              ESPECIAS Y ADEREZOS   57580   80636   91758  108561
##         d5      d6      d7     dd8      d9     d10
## 1  2576134 2593607 2839141 2770198 2740160 2710885
## 2  2965671 3228132 3708675 3943535 4183472 4724145
## 3  1668224 1725576 1783611 1808792 1827177 1982693
## 4  1582291 1783207 1966252 2123150 2360369 3091577
## 5  1031991 1115892 1451119 1540150 2282137 2713540
## 6   405520  404737  451280  418855  398713  365472
## 7   517938  571262  704867  765013  882037 1384251
## 8   202397  190093  157009  173545  164273  163299
## 9   193544  197424  188956  180809  182252  208958
## 10  286507  297299  333812  437266  496656  865432
## 11  229090  214818  214251  224368  221747  228002
## 12  124502  128589  109801  126464  143134  225452
## 13  116499  134123  155394  152145  167650  182256
```

```r
deciles <- deciles %>% column_to_rownames(var  = 'X1')
comp_enigh <- princomp(deciles)
```

Veamos las primeras dos componente, cuyas direcciones principales son:

```r
comp_enigh$loadings[,1:2]
```

```
##         Comp.1      Comp.2
## d1  -0.1224572 -0.29709420
## d2  -0.1858230 -0.35463967
## d3  -0.2324626 -0.34856083
## d4  -0.2610938 -0.29531199
## d5  -0.3010861 -0.23322242
## d6  -0.3221099 -0.16749474
## d7  -0.3650886 -0.07154327
## dd8 -0.3783732  0.02514659
## d9  -0.4019500  0.30553359
## d10 -0.4425357  0.62905737
```

Y los scores son:


```r
comp_enigh$scores[,1:2]
```

```
##                                      Comp.1      Comp.2
## CEREALES                         -4548094.6 -1191716.24
## CARNES                           -7068531.1   392100.57
## PESCADOS Y MARISCOS               1880143.9   290076.15
## LECHE Y SUS DERIVADOS            -2688171.0   541441.01
## HUEVO                             1882277.7  -346789.97
## ACEITES Y GRASAS                  2525107.6  -157761.37
## TUBERCULOS                        2460994.3  -134899.61
## VERDURAS, LEGUMBRES, LEGUMINOSAS -2029622.4  -715856.37
## FRUTAS                             960953.6   445884.20
## AZUCAR Y MIELES                   2551904.3  -217378.43
## CAFE, TE Y CHOCOLATE              2685873.3   -33326.41
## ESPECIAS Y ADEREZOS               2679471.1   -33837.02
## OTROS ALIMENTOS DIVERSOS         -1292306.9  1162063.49
```

Y la tabla de rango 1 es

```r
tab_1 <- tcrossprod(comp_enigh$scores[,1], comp_enigh$loadings[,1])
colnames(tab_1) <- colnames(deciles)
tab_1 <- tab_1 %>% data.frame %>% mutate(categoria = rownames(deciles)) %>%
  gather(decil, gasto, d1:d10)
tab_1$categoria <- reorder(tab_1$categoria, tab_1$gasto, mean)
ggplot(tab_1, aes(x=categoria, y=gasto, colour=decil, group=decil)) +
  geom_line() + coord_flip()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-93-1.png" width="672" />

Que podemos comparar con el análisis no centrado:


```r
svd_enigh <- svd(deciles)
tab_1 <- tcrossprod(svd_enigh$u[,1], svd_enigh$v[,1])
colnames(tab_1) <- colnames(deciles)
tab_1 <- tab_1 %>% data.frame %>% mutate(categoria = rownames(deciles)) %>%
  gather(decil, gasto, d1:d10)
tab_1$categoria <- reorder(tab_1$categoria, tab_1$gasto, mean)
ggplot(tab_1, aes(x=categoria, y=gasto, colour=decil, group=decil)) +
  geom_line() + coord_flip()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-94-1.png" width="672" />


Y aunque los resultados son similares,
puede ser más simple entender la primera dimensión del svd no centrado
que guarda los efectos de los distintos niveles de gasto de
los deciles. En el caso del análisis centrado, tenemos una primera componente
que sólo se entiende bien sabiendo los niveles promedio de gasto
a lo largo de las categorías.

*Observación*: Quizá una solución más natural es hacer el análisis de componentes principales usando la transpuesta de esta matriz (usa la función *prcomp*), donde tiene más sentido
centrar por categoría de alimento, y pensar que las observaciones son los distintos
deciles (que en realidad son agrupaciones de observaciones).


### Otros tipos de centrado

Es posible hacer doble centrado, por ejemplo (por renglón y por columna). Discute por qué
el doble centrado puede ser una buena idea para los datos del tipo de Netflix.

### Reescalando variables

Cuando las columnas tienen distintas unidades (especialmente si las escalas son
muy diferentes), conviene reescalar la matriz antes de hacer el análisis centrado
o no centrado. De otra forma, parte del análisis intenta absorber la diferencia en 
unidades, lo cual generalmente no es de interés.

 - En componentes principales, podemos estandarizar las columnas.
 - En el análisis no centrado, podemos poner las variables en escala 0-1, por ejemplo,
 o dividir entre la media (si son variables positivas).

### Ejemplo {-}


```r
comp <- princomp(attenu %>% select(-station, -event))
comp$loadings[,1]
```

```
##          mag         dist        accel 
## -0.005746131 -0.999982853  0.001129688
```

Y vemos que la dirección de la primera componente es justamente en la dirección de 
la variable *dist* (es decir, la primera componente es *dist*). Esto es porque 
la escala de *dist* es más amplia:


```r
apply(attenu %>% select(-station), 2, mean)
```

```
##      event        mag       dist      accel 
## 14.7417582  6.0840659 45.6032967  0.1542198
```

Esto lo corregimos estandarizando las columnas, o equivalentemente, usando
*cor = TRUE* como opción en *princomp*


```r
comp <- princomp(attenu %>% select(-station, -event), cor  = TRUE)
comp$loadings[,1]
```

```
##        mag       dist      accel 
## -0.5071375 -0.7156080  0.4803298
```


## Otros métodos: t-SNE

Existen otros métodos para reducir dimensionalidad, como MDS (multidimensional
scaling, que se concentra en preservar distancias entre casos), o
métodos de *embeddings* basados en redes neuronales (usar datos en capas
ocultas con un número relativamente chico de unidades, cuando queremos
que nuestra representación esté asociada a una variable respuesta). 

Aquí mostramos una de estas técnicas:  [t-SNE](http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf),
*t-Stochastic Neighbor Embedding*.

Como motivación a esta técnica, observemos primero que
los métodos como componentes principales buscan obtener una representación que mantenga lejos casos que son muy diferentes (pues buscamos aproximar los datos, o encontrar direcciones de máxima varianza). Sin embargo, en algunos
casos los que más nos puede interesar es una representación que **mantenga
casos similares cercanos**, aún cuando perdamos información acerca de distancia de casos muy distintos.


### Ejemplo {-}

Consideremos los siguientes datos:


```r
x_1 <- c(rnorm(100, 0, 0.5), -20, -25, 20, 22)
x_2 <- c(rnorm(50, 1, 0.2), rnorm(50, -1, 0.1), 5, -10, -1, 1)
color <- as.character(c(rep(1, 50), rep(2, 50), rep(3, 4)))
dat_ej <- data_frame(x_1, x_2, color)
ggplot(dat_ej, aes(x = x_1, y = x_2, colour = color)) + geom_point()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-98-1.png" width="480" />

Notamos que la primera dimensión de svd va en dirección del eje 1, 
donde hay más dispersión en los datos alrededor del origen. 


```r
svd(dat_ej[,1:2])$v
```

```
##             [,1]        [,2]
## [1,] -0.99640133  0.08476078
## [2,] -0.08476078 -0.99640133
```

Los datos proyectados, sin embargo, oculta la estructura de grupos que hay
en los puntos cercanos al origen, pues la proyección es


```r
qplot(svd(dat_ej[,1:2])$u[,1], fill = dat_ej$color) + xlab('u')
```

```
## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-100-1.png" width="384" />


Aunque es posible considerar las siguientes aproximaciones, en un problema
de dimensión alta esto puede querer decir que nos costará más trabajo encontrar la estructura de casos similares usando estas técnicas lineales.

En t-SNE construimos una medida de similitud entre puntos que busca concentrarse en puntos similares. Por ejemplo, la estructura de clusters
de estos datos podemos recuperarla en una dimensión:


```r
library(tsne)
sne_ejemplo <- tsne(as.matrix(dat_ej[,1:2]), k = 1, perplexity=100)
qplot(sne_ejemplo, fill = dat_ej$color) + xlab('u')
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-101-1.png" width="384" />


### SNE

t-SNE es una adaptación de SNE (Stochastic Neighbor Embedding). Veamos
primero las ideas básicas de esta técnica:

Para controlar las distancias que nos interesa preservar, primero
introducimos una medida de similitud entre dos casos (renglones)
$x^{(i)}, x^{(j)}$:

\begin{equation}
p_{j|i} = \frac{1}{P_i}\exp\left(- ||x_j - x_i ||^2 / 2\sigma_i^2 \right)
(\#eq:similitud)
\end{equation}


donde $P_i$ es una constante de normalización tal que $\sum_j {p_{j|i}} = 1$.
Por el momento pensemos que la $\sigma_i$ está fija en algún valor arbitrario.

- Notamos que $p_{j|i}$ toma valores cercanos a 1 cuando 
$x_i$ y $x_j$ están cercanos (su distancia es chica), y rápidamente
decae a 0 cuando $x_i$ y $x_j$ se empiezan a alejar.
- Cuando la $\sigma$ es chica, si $x_i$ y $x_j$ están aunque sea un poco
separados, $p_{j|i}\approx 0$. Si $\sigma$ es grande, entonces esta
cantidad decae más lentamente cuando los puntos se alejan
- Podemos pensar que tenemos una campana gaussiana centrada en $x_i$, y
la $p_{j|i}$ está dada por la densidad mostrada arriba evaluada en $x_j$

Ahora pensemos que buscamos convertir estos dos puntos a un nuevo
espacio de dimensión menor. Denotamos por $y_j$ y $y_i$ a los puntos
correpondientes en el nuevo espacio. Definimos análogamente :

$$q_{j|i} = \frac{1}{Q_i}\exp\left(- ||y_j - y_i ||^2  \right)$$

**¿Cómo encontramos los valores $y_j$?** La idea es intentar aproximar las similitudes derivadas:

$$ p_{j|i} \approx q_{j|i},$$

Pensemos que $i$ está fija (que corresponden a los puntos $x^{(i)}$ y
$y^{(i)}$). Lo que queremos hacer es **aproximar la estructura local** alrededor de $x^{(i)}$ mediante los puntos $y$. Y esto queremos hacerlo
para cada caso $i$ en los datos.

 Nótese que estas similitudes son sensibles cuando los
puntos están relativamente cerca, pero se hacen rápidamente cero cuando
las distancias son más grandes. Esto permite que al intentar hacer la 
aproximación realmente nos concentremos en la estructura local alrededor de cada punto $x^{(i)}$

- Nótese que usamos $\sigma=1$ en la definición de las $q_{j|i}$, pues esto depende de la escala de la nueva representación (que vamos a encontrar).
- Dependiendo de la $\sigma_i$, podemos afinar el método para definir qué
tan local es la estructura que queremos aproximar.


### Minimización para SNE

En SNE, el objetivo a minimizar es la divergencia de
Kullback-Liebler, que es una especie de 
distancia entre distribuciones de probabilidad. Buscamos
resolver

$$\min_{y^{1}, y^{2}, \ldots, y^{(n)}} \sum_{i,j} p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}$$

Existen muchas maneras de entender esta cantidad. Por lo pronto,
notemos que 

- Si $p_{j|i} = q_{1|j}$, entonces esta cantidad es 0 
(pues el logaritmo de 1 es cero). 
- Para cada $i$,  $\sum_{j} p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}\geq 0$. Demuestra usando cálculo.
- Y **más importante**:  $p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}$ es muy grande
cuando obtenemos una $q_{j|i}$ chica para una $p_{j|i}$ grande (representar lejos puntos cercanos), pero no aumenta
tanto si obtenemos una $q_{j|i}$ grande para una $p_{j|i}$ chica. Puedes
usar cálculo para convencerte de esto también, por ejemplo, derivando
con respecto a las $x$ la cantidad $\sum_i p_i \log p_i/x_i -\sum_i x_i$
 (donde $q_i = x_i/\sum_i x_i$, $\sum_i x_i$ es la constante de normalización). 

Finalmente, la **divergencia se minimiza usando descenso en gradiente**.

t-SNE se basa en las mismas ideas, con algunas mejoras (ver [paper](http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)): 

- Utiliza una
versión simétrica de la distancia de Kullback-Liebler (que simplifica
el cálculo del gradiente para obtener un algoritmo más rápido), y utiliza
una distribución t en el espacio de baja dimensión en lugar de una guassiana. Esto último mejora el desempeño del algoritmo en dimensiones altas.


### Ejemplo {-}
Uno de los ejemplos clásicos es con imágenes de dígitos


```r
set.seed(288022)
zip_train <- read_csv('datos/zip-train.csv')
muestra_dig <- zip_train %>% sample_n(500)
tsne_digitos <- tsne(muestra_dig[,2:257], max_iter=500)
```


```r
dat_tsne <- data.frame(tsne_digitos)
dat_tsne$digito <- as.character(muestra_dig$X1)
ggplot(dat_tsne, aes(x=X1, y=X2, colour=digito, label=digito)) + geom_text()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-103-1.png" width="672" />

Comparemos con componentes principales, que no logra separar muy
bien los distintos dígitos:

```r
comps <- princomp(muestra_dig[,-1])
dat_comps_tsne <- data.frame(comps$scores[,1:2])
dat_comps_tsne$digito <- as.character(muestra_dig$X1)
ggplot(dat_comps_tsne, aes(x=Comp.1, y=Comp.2, colour=digito, label=digito)) + geom_text()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-104-1.png" width="672" />

y vemos, por ejemplo, que la primera componente está separando
principalmente ceros de unos. La primera dirección principal es:


```r
image(matrix(comps$loadings[,1], 16,16))
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-105-1.png" width="384" />


### Perplexity

El único parámetro que resta discutir es $\sigma_i$ en 
\@ref(eq:similitud). La primera idea es que distintas regiones
(vecindades de $x^{(i)}$) pueden requerir distintas $\sigma_i$. Este
valor se escoge de forma que todos los puntos tengan aproximadamente un
número fijo (aproximado) de vecinos, a través de un valor que se llama 
*perplexity*. De esta forma, $\sigma_i$ en regiones donde hay densidad
alta de datos es más chica, y $sigma_i$ en regiones donde hay menos densidad
es más grande (para mantener el número de vecinos aproximadamente similar).

Antes de ver más detalles, veamos el ejemplo de los dígitos cambiando
el valor de perplejidad:


```r
tsne_digitos <- tsne(muestra_dig[,2:257], perplexity = 5,
                     max_iter=500)
dat_tsne <- data.frame(tsne_digitos)
dat_tsne$digito <- as.character(muestra_dig$X1)
ggplot(dat_tsne, aes(x=X1, y=X2, colour=digito, label=digito)) + geom_text()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-106-1.png" width="672" />


```r
tsne_digitos <- tsne(muestra_dig[,2:257], perplexity = 100,
                     max_iter=500)
dat_tsne <- data.frame(tsne_digitos)
dat_tsne$digito <- as.character(muestra_dig$X1)
ggplot(dat_tsne, aes(x=X1, y=X2, colour=digito, label=digito)) + geom_text()
```

<img src="14-reducir-dimensionalidad_files/figure-html/unnamed-chunk-107-1.png" width="672" />


**Notas** (opcional):
si $P$ es el valor de perplejidad escogido,  el valor de $\sigma_i$ 
escoge de manera que las $p_{j|i}$ obtenidas satisfagan
$2^{-\sum_{j} p_{j|i}\log_2 p_{j|i}}\approx P$. El valor de la izquierda en
esta ecuación se puede interpretar como una medida suave del número
de puntos donde se concentra la masa de probabilidad. Por ejemplo:


```r
perp <- function(x){2^sum(-x*log2(x))}
p <- c(0.01, 0.01, 0.01, 0.02, 0.95)
sum(p)
```

```
## [1] 1
```

```r
perp(p)
```

```
## [1] 1.303593
```

```r
p <- c(0.01, 0.18, 0.01, 0.4, 0.4 )
perp(p)
```

```
## [1] 3.107441
```

```r
p <- c(rep(1,20), rep(20,20))
perp(p/sum(p))
```

```
## [1] 24.21994
```



### Tarea (para 27 de Noviembre) {-}

1. Validación:  lee este [blogpost](http://www.fast.ai/2017/11/13/validation-sets/), acerca
de la construcción apropiada de conjuntos de validación y prueba.

2. Reducción de dimensionalidad: ve *script/tarea_12_dimensionalidad.Rmd*

### Tarea (para 4 de diciembre)

Prepara con tu equipo una descripción corta (1-2 párrafos describiendo
objetivo, datos y métodos) de lo que piensan hacer como trabajo final.
 
Las condiciones para el examen final son:

- La presentación final será de unos 7 minutos
máximo (se penalizará pasarse del tiempo). Pueden usar un documento
de html, pdf o diapositivas para presentar.

- En caso de ser necesario, les pediré a los equipos el documento presentado
con posibles preguntas adicionales.

- La calificación se hará en tres dimensiones: complejidad y tratamiento de datos (datos complejos y bien procesados dan más puntos), ejecución
de los modelos (correcta selección de parámetros, validación), y 
presentación (explicaciones claras de puntos importantes).

- Sugerencias: escoger un método o variación de métodos que no hayamos visto en clase, y aplicarlo a unos cuantos ejemplos de datos. También es posible concentrarse en un conjunto de datos y aplicar algunos métodos para obtener
las mejores predicciones posibles. Recuerden que es mejor hacer un proyecto relativamente limitado con explicaciones y resultados claros que un proyecto
demasiado complejo que no puedan explicar razonablemente bien en el tiempo
que tienen.

