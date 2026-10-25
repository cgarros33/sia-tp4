# Alternativa: actualizar dos hiperparámetros en tándem (sweep4, k=4)

Se comparó el acoplamiento `radio_inicial_relativo × decaimiento_radio` de
`sweep4` contra la configuración secuencial.

| Configuración | QE media ± desvío | TE | Unidades muertas | Ocupación |
|---|---:|---:|---:|---:|
| Secuencial: radio 0.75, decaimiento lineal | 3.180673 ± 0.021266 | 0.0333 | 0.33 | 0.9792 |
| Tándem: radio 1.0, decaimiento inversa | 3.173119 ± 0.013589 | 0.0467 | 1.00 | 0.9375 |

La mejora de QE es **0.007554 absoluta (0.2375%)**. El acoplamiento
`eta_inicial × decaimiento_eta` no mejoró: eligió `(0.9, lineal)`, igual que la
configuración vigente.

## Comparación por semilla

- Diferencias pareadas (secuencial − tándem): `[0, 0.001057, 0.021604]`.
- t pareada ≈ 1.07 (gl = 2), p ≈ 0.40.
- IC 95% de la mejora: `[-0.0227, +0.0378]`, incluye el cero.
- Solo **2 de 24** combinaciones de radio × decaimiento superan la QE
  secuencial: `(0.75, inversa)` y `(1.0, inversa)`.

## Actualización secuencial equivalente

1. Cambiar solo el decaimiento a `inversa`: QE **3.175807** (mejora 0.004865).
2. Cambiar después solo el radio a `1.0`: QE **3.173119** (mejora adicional
   0.002689).

Se llega al mismo óptimo sin acoplar los dos valores. El tándem solo evita el
efecto de reevaluar con valores desactualizados; no aporta una mejora real.

## Conclusión

Bajo la regla de menor QE media, el acoplamiento se adopta, pero la ganancia es
marginal y está dentro del ruido de las 3 semillas. Además, empeora TE
(0.0333 → 0.0467), unidades muertas (0.33 → 1.00) y ocupación
(0.9792 → 0.9375). Para `sweep4`, actualizar en tándem no ofrece un beneficio
práctico frente a hacerlo uno después del otro.
