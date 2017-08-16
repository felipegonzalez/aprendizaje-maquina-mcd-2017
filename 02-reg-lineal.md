# Regresión lineal {#regresion}



## Introducción

Consideramos un problema de regresión con entradas $X=(X_1,X_2,\ldots, X_p)$
y salida $Y$. Una de las maneras más simples que podemos intentar
para predecir $Y$ en función de las $X_j$´s es mediante una suma ponderada
de los valores de las $X_j's$, usando una función

$$f_\beta (X) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p,$$
Nuestro trabajo será entonces, dada una muestra de entrenamiento ${\mathcal L}$,
encontrar valores apropiados de las $\beta$'s, para construir un predictor:

$$\hat{f}(X) = \hat{\beta}_0 + \hat{\beta}_1 X_1 + \hat{\beta}_2 X_2 \cdots + \hat{\beta} X_p$$
y usaremos esta función $\hat{f}$ para hacer predicciones $\hat{Y} =\hat{f}(X)$.



#### Ejemplos
Queremos predecir las ventas futuras anuales $Y$ de un supermercado que se va a construir
en un lugar dado. Las variables que describen el lugar son
$X_1 = trafico\_coches$, $X_2=trafico\_peatones$. En una aproximación simple,
podemos suponer que la tienda va a capturar una fracción de esos tráficos que
se van a convertir en ventas. Quisieramos predecir con una función de la forma
$$f_\beta (coches, peatones) = \beta_0 + \beta_1\, coches + \beta_2\, peatones.$$
Por ejemplo, después de un análisis estimamos que 

- $\hat{\beta}_0 = 1000000$ (ventas base)
- $\hat{\beta}_1 = (200)*0.02 = 4$
- $\hat{\beta}_2 = (300)*0.01 =3$

Entonces haríamos predicciones con
$$\hat{f}(peatones, coches) = 1000000 +  4\,peatones + 3\, coches$$

El modelo lineal es más flexible de lo que parece en una primera aproximación, porque
tenemos libertad para construir las variables de entrada a partir de nuestros datos.
Por ejemplo, si tenemos una tercera variable 
$estacionamiento$ que vale 1 si hay un estacionamiento cerca o 0 si no lo hay, podríamos
definir las variables

- $X_1= peatones$
- $X_2 = coches$
- $X_3 = estacionamiento$
- $X_4 = coches*estacionamiento$

Donde la idea de agregar $X_4$ es que si hay estacionamiento entonces vamos
a capturar una fracción adicional del trafico de coches, y la idea de $X_3$ es que 
la tienda atraerá más nuevas visitas si hay un estacionamiento cerca. Buscamos 
ahora modelos de la forma

$$f_\beta(X_1,X_2,X_3,X_4) = \beta_0 + \beta_1X_1 + \beta_2 X_2 + \beta_3 X_3 +\beta_4 X_4$$

y podríamos obtener después de nuestra análisis las estimaciones


- $\hat{\beta}_0 = 800000$ (ventas base)
- $\hat{\beta}_1 = 4$
- $\hat{\beta}_2 = (300)*0.005 = 1.5$
- $\hat{\beta}_3 = 400000$
- $\hat{\beta}_4 = (300)*0.01 = 3$
 
 y entonces haríamos predicciones con el modelo

$$\hat{f} (X_1,X_2,X_3,X_4) = 
800000 + 4\, X_1 + 1.5 \,X_2 + 400000\, X_3 +3\, X_4$$

## Aprendizaje de coeficientes (ajuste)

En el ejemplo anterior, los coeficientes fueron calculados (o estimados) usando
experiencia, argumentos teóricos, o quizá otras fuentes de datos (como estudios
o encuestas, conteos, etc.) 

Ahora quisiéramos construir un algoritmo para
aprender estos coeficientes del modelo

$$f_\beta (X_1) = \beta_0 + \beta_1 X_1 + \cdots \beta_p X_p$$
a partir de una muestra de entrenamiento

$${\mathcal L}=\{ (x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}), \ldots, (x^{(N)}, y^{(N)}) \}$$

El criterio de ajuste (algoritmo de aprendizaje) más usual para regresión 
lineal es el de **mínimos cuadrados**. 



Construimos las predicciones (ajustados) para la muestra de entrenamiento:
$$\hat{y}^{(i)} =  f_\beta (x^{(i)}) = \beta_0 + \beta_1 x_1^{(i)}+ \cdots + \beta_p x_p^{(i)}$$

Y consideramos las diferencias de los ajustados con los valores observados:

$$e^{(i)} = y^{(i)} - f_\beta (x^{(i)})$$

La idea entonces es minimizar la suma de los residuales al cuadrado, para
intentar que la función ajustada pase lo más cercana a los puntos de entrenamiento 
que sea posible. Si

$$RSS(\beta) = \sum_{i=1}^N (y^{(i)} - f_\beta(x^{(i)}))^2$$
Queremos resolver

<div class="comentario">
<p><strong>Mínimos cuadrados</strong></p>
<p><br /><span class="math display">$$\min_{\beta} RSS(\beta) = \min_{\beta}\sum_{i=1}^N (y^{(i)} - f_\beta(x^{(i)}))^2$$</span><br /></p>
</div>

#### Ejemplo

Consideremos 


```r
library(readr)
library(dplyr)
library(knitr)
prostata <- read_csv('datos/prostate.csv') %>% select(lcavol, lpsa, train)
kable(head(prostata), format = 'html')
```

<table>
 <thead>
  <tr>
   <th style="text-align:right;"> lcavol </th>
   <th style="text-align:right;"> lpsa </th>
   <th style="text-align:left;"> train </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> -0.5798185 </td>
   <td style="text-align:right;"> -0.4307829 </td>
   <td style="text-align:left;"> TRUE </td>
  </tr>
  <tr>
   <td style="text-align:right;"> -0.9942523 </td>
   <td style="text-align:right;"> -0.1625189 </td>
   <td style="text-align:left;"> TRUE </td>
  </tr>
  <tr>
   <td style="text-align:right;"> -0.5108256 </td>
   <td style="text-align:right;"> -0.1625189 </td>
   <td style="text-align:left;"> TRUE </td>
  </tr>
  <tr>
   <td style="text-align:right;"> -1.2039728 </td>
   <td style="text-align:right;"> -0.1625189 </td>
   <td style="text-align:left;"> TRUE </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 0.7514161 </td>
   <td style="text-align:right;"> 0.3715636 </td>
   <td style="text-align:left;"> TRUE </td>
  </tr>
  <tr>
   <td style="text-align:right;"> -1.0498221 </td>
   <td style="text-align:right;"> 0.7654678 </td>
   <td style="text-align:left;"> TRUE </td>
  </tr>
</tbody>
</table>

```r
prostata_entrena <- filter(prostata, train)
ggplot(prostata_entrena, aes(x = lcavol, y = lpsa)) + geom_point()
```

<img src="02-reg-lineal_files/figure-html/unnamed-chunk-3-1.png" width="384" />

En este caso, buscamos ajustar el modelo (tenemos una sola entrada)
$f_{\beta} (X_1) = \beta_0 + \beta_1 X_1$,
que es una recta. Los cálculos serían como sigue:


```r
rss_calc <- function(datos){
  y <- datos$lpsa
  x <- datos$lcavol
  fun_out <- function(beta){
    y_hat <- beta[1] + beta[2]*x
    e <- (y - y_hat)
    rss <- sum(e^2)
    0.5*rss
  }
  fun_out
}
```

Nuestra función rss es entonces:


```r
rss <- rss_calc(prostata_entrena)
```

Por ejemplo, si consideramos $(\beta_0, \beta_1) = (1, 1)$, obtenemos


```r
beta <- c(0,1.5)
rss(beta)
```

```
## [1] 61.63861
```
Que corresponde a la recta


```r
ggplot(prostata_entrena, aes(x = lcavol, y = lpsa)) + geom_point() +
  geom_abline(slope = beta[2], intercept = beta[1], col ='red')
```

<img src="02-reg-lineal_files/figure-html/unnamed-chunk-7-1.png" width="384" />

Podemos comparar con  $(\beta_0, \beta_1) = (1, 1)$, obtenemos


```r
beta <- c(1,1)
rss(beta)
```

```
## [1] 27.11781
```

```r
ggplot(prostata_entrena, aes(x = lcavol, y = lpsa)) + geom_point() +
  geom_abline(slope = beta[2], intercept = beta[1], col ='red')
```

<img src="02-reg-lineal_files/figure-html/unnamed-chunk-8-1.png" width="384" />

Ahora minimizamos. Podríamos hacer


```r
res_opt <- optim(c(0,0), rss, method = 'BFGS')
beta_hat <- res_opt$par
beta_hat
```

```
## [1] 1.5163048 0.7126351
```

```r
res_opt$convergence
```

```
## [1] 0
```


```r
ggplot(prostata_entrena, aes(x = lcavol, y = lpsa)) + geom_point() +
  geom_abline(slope = 1, intercept = 1, col ='red') +
  geom_abline(slope = beta_hat[2], intercept = beta_hat[1]) 
```

<img src="02-reg-lineal_files/figure-html/unnamed-chunk-10-1.png" width="672" />



## Descenso en gradiente

Aunque el problema de mínimos cuadrados se puede resolver analíticamente, proponemos
un método numérico básico que es efectivo y puede escalarse a problemas grandes
de manera relativamente simple: descenso en gradiente, o descenso máximo.

Supongamos que una función $h(x)$ es convexa y tiene un mínimo. La idea
de descenso en gradiente es comenzar con un candidato inicial $z_0$ y calcular
la derivada en $z^{(0)}$. Si $h(z^{(0)})<0$, la función es creciente en $z^{(0)}$ y nos
movemos ligeramente 
a la izquierda para obtener un nuevo candidato $z^{(1)}$. si $h(z^{(0)})<0$, la
función es decreciente en $z^{(0)}$ y nos
movemos ligeramente a la derecha  para obtener un nuevo candidato $z^{(1)}$. Iteramos este
proceso hasta que la derivada es cercana a cero (estamos cerca del óptimo).

Si $\eta>0$ es una cantidad chica, podemos escribir

$$z^{(1)} = z^{(0)} + \eta \,h'(z^{(0)}).$$

Nótese que cuando la derivada tiene magnitud alta, el movimiento de $z^{(0)}$ a $z^{(1)}$
es más grande, y siempre nos movemos una fracción de la derivada. En general hacemos
$$z^{(j+1)} = z^{(j)} + \eta\,h'(z^{(j)})$$
para obtener una sucesión $z^{(0)},z^{(1)},\ldots$. Esperamos a que $z^{(j)}$ converja
para terminar la iteración.



#### Ejemplo

Si tenemos

```r
 h <- function(x) {
x^2 + (x - 2)^2 - log(x^2 + 1)
 }
```

Calculamos (a mano):

```r
 h_deriv <- function(x) {
  2 * x + 2 * (x - 2) - 2*x/(x^2 + 1)
 }
```

Ahora iteramos con $\eta = 0.1$ y valor inicial $z_0=5$

```r
z_0 <- 5
eta <- 0.4
descenso <- function(h, h_deriv){
  fun <- function(n, z_0, eta){
    z <- matrix(0,n, length(z_0))
    z[1, ] <- z_0
    for(i in 1:(n-1)){
      z[i+1, ] <- z[i, ] - eta * h_deriv(z[i, ])
    }
    z
  }
  fun
}
descenso_fun <- descenso(h, h_deriv) 
z <- descenso_fun(20, 5, 0.1)
z
```

```
##           [,1]
##  [1,] 5.000000
##  [2,] 3.438462
##  [3,] 2.516706
##  [4,] 1.978657
##  [5,] 1.667708
##  [6,] 1.488834
##  [7,] 1.385872
##  [8,] 1.326425
##  [9,] 1.291993
## [10,] 1.272002
## [11,] 1.260375
## [12,] 1.253606
## [13,] 1.249663
## [14,] 1.247364
## [15,] 1.246025
## [16,] 1.245243
## [17,] 1.244788
## [18,] 1.244523
## [19,] 1.244368
## [20,] 1.244277
```

Y vemos que estamos cerca de la convergencia.


```r
curve(h, -3, 6)
points(z[,1], h(z))
text(z[1:6], h(z[1:6]), pos = 3)
```

<img src="02-reg-lineal_files/figure-html/unnamed-chunk-14-1.png" width="480" />



### Selección de tamaño de paso $\eta$

Si hacemos $\eta$ muy chico, el algoritmo puede tardar mucho en
converger:

```r
z <- descenso_fun(20, 5, 0.01)
curve(h, -3, 6)
points(z, h(z))
text(z[1:6], h(z[1:6]), pos = 3)
```

<img src="02-reg-lineal_files/figure-html/unnamed-chunk-15-1.png" width="384" />

Si hacemos $\eta$ muy grande, el algoritmo puede divergir:


```r
z <- descenso_fun(20, 5, 1.5)
z
```

```
##                [,1]
##  [1,]  5.000000e+00
##  [2,] -1.842308e+01
##  [3,]  9.795302e+01
##  [4,] -4.837345e+02
##  [5,]  2.424666e+03
##  [6,] -1.211733e+04
##  [7,]  6.059265e+04
##  [8,] -3.029573e+05
##  [9,]  1.514792e+06
## [10,] -7.573955e+06
## [11,]  3.786978e+07
## [12,] -1.893489e+08
## [13,]  9.467445e+08
## [14,] -4.733723e+09
## [15,]  2.366861e+10
## [16,] -1.183431e+11
## [17,]  5.917153e+11
## [18,] -2.958577e+12
## [19,]  1.479288e+13
## [20,] -7.396442e+13
```

<div class="comentario">
<p>Es necesario ajustar el tamaño de paso para cada problema particular. Si la convergencia es muy lenta, podemos incrementarlo. Si las iteraciones divergen, podemos disminuirlo</p>
</div>

### Funciones de varias variables
Si ahora $h(z)$ es una función de $p$ variables, podemos intentar
la misma idea usando el gradiente. Por cálculo sabemos que el gradiente
apunta en la dirección de máximo crecimiento local. El gradiente es
el vector columna con las derivadas parciales de $h$:

$$\nabla h(z) = \left( \frac{\partial h}{\partial z_1}, \frac{\partial h}{\partial z_2}, \ldots,    \frac{\partial h}{\partial z_p} \right)^t$$
Y el paso de iteración, dado un valor inicial $z_0$ y un tamaño de paso
$\eta >0$ es

$$z^{(i+1)} = z^{(i)} - \eta \nabla h(z^{(i)})$$

Las mismas consideraciones acerca del tamaño de paso $\eta$ aplican en
el problema multivariado.


```r
h <- function(z) {
  z[1]^2 + z[2]^2 - z[1] * z[2]
}
h_gr <- function(z_1,z_2) apply(cbind(z_1, z_2), 1, h)
grid_graf <- expand.grid(z_1 = seq(-3, 3, 0.1), z_2 = seq(-3, 3, 0.1))
grid_graf <- grid_graf %>%  mutate( val = apply(cbind(z_1,z_2), 1, h))
gr_contour <- ggplot(grid_graf, aes(x = z_1, y = z_2, z = val)) + 
  geom_contour(binwidth = 1.5, aes(colour = ..level..))
gr_contour
```

<img src="02-reg-lineal_files/figure-html/unnamed-chunk-18-1.png" width="672" />

El gradiente está dado por


```r
h_grad <- function(z){
  c(2*z[1] - z[2], 2*z[2] - z[1])
}
```

Podemos graficar la dirección de máximo descenso para diversos puntos. Estas
direcciones son ortogonales a la curva de nivel que pasa por cada uno de los
puntos:


```r
grad_1 <- h_grad(c(0,-2))
grad_2 <- h_grad(c(1,1))
eta <- 0.2
#library(grid)
gr_contour +
  geom_segment(aes(x=0.0, xend=0.0-eta*grad_1[1], y=-2,
     yend=-2-eta*grad_1[2]),
    arrow = arrow(length = unit(0.2,"cm")))+ 
  geom_segment(aes(x=1, xend=1-eta*grad_2[1], y=1,
     yend=1-eta*grad_2[2]),
    arrow = arrow(length = unit(0.2,"cm")))+ coord_fixed(ratio = 1)
```

<img src="02-reg-lineal_files/figure-html/unnamed-chunk-20-1.png" width="672" />

Y aplicamos descenso en gradiente:


```r
des_h <- descenso(h, h_grad)
des_h(20, c(1,1), 0.1)
```

```
##            [,1]      [,2]
##  [1,] 1.0000000 1.0000000
##  [2,] 0.9000000 0.9000000
##  [3,] 0.8100000 0.8100000
##  [4,] 0.7290000 0.7290000
##  [5,] 0.6561000 0.6561000
##  [6,] 0.5904900 0.5904900
##  [7,] 0.5314410 0.5314410
##  [8,] 0.4782969 0.4782969
##  [9,] 0.4304672 0.4304672
## [10,] 0.3874205 0.3874205
## [11,] 0.3486784 0.3486784
## [12,] 0.3138106 0.3138106
## [13,] 0.2824295 0.2824295
## [14,] 0.2541866 0.2541866
## [15,] 0.2287679 0.2287679
## [16,] 0.2058911 0.2058911
## [17,] 0.1853020 0.1853020
## [18,] 0.1667718 0.1667718
## [19,] 0.1500946 0.1500946
## [20,] 0.1350852 0.1350852
```

```r
 ggplot(data= grid_graf) + 
  geom_contour(binwidth = 1.5, aes(x = z_1, y = z_2, z = val, colour = ..level..)) + 
   geom_point(data = data.frame(des_h(20, c(3,1), 0.3)), aes(x=X1, y=X2), colour = 'red')
```

<img src="02-reg-lineal_files/figure-html/unnamed-chunk-21-1.png" width="672" />


## Descenso en gradiente para regresión lineal

Vamos a escribir ahora el algoritmo de descenso en gradiente para regresión lineal.
Igual que en los ejemplos anteriores, tenemos que precalcular el gradiente. Una
vez que esto esté terminado, escribir la iteración es fácil.

Recordamos que queremos minimizar (dividiendo entre dos para simplificar más adelante)
$$RSS(\beta) = \frac{1}{2}\sum_{i=1}^N (y^{(i)} - f_\beta(x^{(i)}))^2$$

La derivada de la suma es la suma de las derivadas, así nos concentramos
en derivar uno de los términos

$$  \frac{1}{2}(y^{(i)} - f_\beta(x^{(i)}))^2 $$
Usamos la regla de la cadena para obtener
$$ \frac{1}{2}\frac{\partial}{\partial \beta_j} (y^{(i)} - f_\beta(x^{(i)}))^2 =
-(y^{(i)} - f_\beta(x^{(i)})) \frac{\partial f_\beta(x^{(i)})}{\partial \beta_j}$$

Tenemos dos casos

$$\frac{\partial f_\beta(x^{(i)})}{\partial \beta_0} = 1$$
y  si $j=1,2,\ldots, p$ entonces

$$\frac{\partial f_\beta(x^{(i)})}{\partial \beta_j} = x_j^{(i)}$$

<div class="comentario">
<p>De modo que <br /><span class="math display">$$\frac{\partial f_\beta(x^{(i)})}{\partial \beta_0} = -(y^{(i)} - f_\beta(x^{(i)}))$$</span><br /> y</p>
<p><br /><span class="math display">$$\frac{\partial f_\beta(x^{(i)})}{\partial \beta_j} = - x_j^{(i)}(y^{(i)} - f_\beta(x^{(i)}))$$</span><br /></p>
</div>

Y sumando todos los términos (uno para cada caso de entrenamiento):

$$\frac{\partial RSS(\beta)}{\partial \beta_0} = - \sum_{i=1}^N e^{(i)} $$
$$\frac{\partial RSS(\beta)}{\partial \beta_j} = - \sum_{i=1}^N x_j^{(i)}e^{(i)} $$

para $j=1,2,\ldots, p$.

Podemos implementar ahora estos cálculos:


```r
grad_calc <- function(x_ent, y_ent){
  salida_grad <- function(beta){
    f_beta <- as.matrix(cbind(1, x_ent)) %*% beta
    e <- y_ent - f_beta
    grad_out <- -apply(t(cbind(1,x_ent)) %*% e, 1, sum)
    names(grad_out)[1] <- 'Intercept'
    grad_out
  }
  salida_grad
}
grad <- grad_calc(prostata_entrena[, 1, drop = FALSE], prostata_entrena$lpsa)
grad(c(0,1))
```

```
## Intercept    lcavol 
## -76.30319 -70.93938
```

```r
grad(c(1,1))
```

```
## Intercept    lcavol 
## -9.303187 17.064556
```

Podemos checar nuestro cálculo del gradiente:

```r
delta <- 0.001
(rss(c(1 + delta,1)) - rss(c(1,1)))/delta
```

```
## [1] -9.269687
```

```r
(rss(c(1,1+delta)) - rss(c(1,1)))/delta
```

```
## [1] 17.17331
```

Y ahora iteramos


```r
descenso_prost <- descenso(rss, grad)
iteraciones <- descenso_prost(20, c(0,0), 0.005)
iteraciones
```

```
##            [,1]      [,2]
##  [1,] 0.0000000 0.0000000
##  [2,] 0.8215356 1.4421892
##  [3,] 0.7332652 0.9545169
##  [4,] 0.8891507 1.0360252
##  [5,] 0.9569494 0.9603012
##  [6,] 1.0353555 0.9370937
##  [7,] 1.0977074 0.9046239
##  [8,] 1.1534587 0.8800287
##  [9,] 1.2013557 0.8576489
## [10,] 1.2430547 0.8385314
## [11,] 1.2791967 0.8218556
## [12,] 1.3105688 0.8074114
## [13,] 1.3377869 0.7948709
## [14,] 1.3614051 0.7839915
## [15,] 1.3818983 0.7745509
## [16,] 1.3996803 0.7663595
## [17,] 1.4151098 0.7592518
## [18,] 1.4284979 0.7530844
## [19,] 1.4401148 0.7477329
## [20,] 1.4501947 0.7430895
```

Y checamos que efectivamente el error total de entrenamiento decrece

```r
apply(iteraciones, 1, rss)
```

```
##  [1] 249.60960  51.70986  32.49921  28.96515  27.22475  25.99191  25.07023
##  [8]  24.37684  23.85483  23.46181  23.16591  22.94312  22.77538  22.64910
## [15]  22.55401  22.48242  22.42852  22.38794  22.35739  22.33438
```

## Normalización de entradas

## Interpretación de modelos lineales

## Solución analítica

## ¿Por qué el modelo lineal funciona bien (muchas veces)?


