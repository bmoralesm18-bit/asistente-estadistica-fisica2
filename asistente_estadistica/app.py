import math
import random
import re
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# DATOS DEL PROYECTO (cámbialos aquí cuando quieras)
# ---------------------------------------------------------------
NOMBRE_APP = "Asistente de Estadística para Física 2"
INTEGRANTES = "Leonardo Morales y Carlos Gabriel Aquino Natareno"
ASIGNATURA = "Física 2"
SECCION = "A"

st.set_page_config(page_title=NOMBRE_APP, page_icon="📊")


# ---------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------
def leer_datos(texto):
    """Convierte el texto escrito por el usuario en una lista de números.
    Devuelve (lista_de_numeros, lista_de_errores)."""
    piezas = [p for p in re.split(r"[,;\s]+", texto.strip()) if p]
    datos = []
    errores = []
    for p in piezas:
        try:
            valor = float(p)
            if not np.isfinite(valor):  # rechaza nan e inf
                raise ValueError
            datos.append(valor)
        except ValueError:
            errores.append(p)
    return datos, errores


def lista_corta(valores, decimales=2):
    """Muestra los valores separados por ' + ' (o '...' si son muchos)."""
    textos = [f"{v:.{decimales}f}" for v in valores]
    if len(textos) > 10:
        textos = textos[:3] + ["..."] + textos[-2:]
    return " + ".join(textos)


def cdf_normal(z):
    """Φ(z): probabilidad acumulada de la normal estándar, P(Z < z).
    Usa la función error de Python, que da el mismo valor que la tabla Z."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def pedir_datos(clave, ejemplo, con_boton=True):
    """Dibuja el cuadro de mediciones y la unidad, y valida lo escrito.
    Devuelve (datos, unidad) si todo está bien, o None si falta algo.
    'clave' evita que dos módulos compartan los mismos cuadros."""
    col1, col2 = st.columns([3, 1])
    with col1:
        texto = st.text_area(
            "Mediciones", value=ejemplo, height=100, key=f"{clave}_texto"
        )
    with col2:
        unidad = st.text_input(
            "Unidad", value="A", help="Ejemplos: A, V, m, s, Ω", key=f"{clave}_unidad"
        )

    if con_boton and not st.button("Calcular", type="primary", key=f"{clave}_boton"):
        return None

    datos, errores = leer_datos(texto)
    if errores:
        st.error(
            "Hay valores que no son números: " + ", ".join(f"'{e}'" for e in errores)
            + ". Corrígelos y vuelve a calcular."
        )
        return None
    if len(datos) == 0:
        st.warning("No ingresaste ninguna medición. Escribe al menos 2 valores.")
        return None
    return datos, unidad


# ---------------------------------------------------------------
# MÓDULO 1: ESTADÍSTICA DESCRIPTIVA + DATOS EXPERIMENTALES
# ---------------------------------------------------------------
def modulo_descriptiva():
    st.header("Estadística descriptiva")
    st.write(
        "Escribe tus mediciones separadas por comas, espacios o saltos de línea. "
        "Usa punto como separador decimal (por ejemplo 2.1)."
    )

    entrada = pedir_datos("descriptiva", "2.1, 2.3, 2.2, 2.4, 2.0")
    if entrada is None:
        return
    datos, unidad = entrada

    if len(datos) == 1:
        st.warning(
            "Con un solo dato no se puede calcular la desviación estándar "
            "muestral (se divide entre n − 1 = 0). Ingresa al menos 2 mediciones."
        )
        return

    # ----- Cálculos -----
    x = np.array(datos)
    n = len(x)
    suma = x.sum()
    media = suma / n
    mediana = float(np.median(x))
    modas = statistics.multimode(datos)
    rango = x.max() - x.min()
    desviaciones = x - media
    suma_cuadrados = float((desviaciones ** 2).sum())
    var_muestral = suma_cuadrados / (n - 1)
    var_poblacional = suma_cuadrados / n
    s = var_muestral ** 0.5          # desviación estándar muestral
    sigma = var_poblacional ** 0.5   # desviación estándar poblacional
    incert_media = s / n ** 0.5      # incertidumbre estándar de la media

    # ----- Resultados finales -----
    st.subheader("Resultados")
    c1, c2, c3 = st.columns(3)
    c1.metric("Media", f"{media:.4f} {unidad}")
    c2.metric("Mediana", f"{mediana:.4f} {unidad}")
    if len(modas) == len(set(datos)):
        c3.metric("Moda", "No hay")
    else:
        c3.metric("Moda", ", ".join(f"{m:g}" for m in modas) + f" {unidad}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Rango", f"{rango:.4f} {unidad}")
    c5.metric("Desv. muestral (s)", f"{s:.4f} {unidad}")
    c6.metric("Desv. poblacional (σ)", f"{sigma:.4f} {unidad}")

    # ----- Procedimiento paso a paso -----
    st.subheader("Procedimiento paso a paso")

    st.markdown("**1. Media aritmética**")
    st.latex(r"\bar{x} = \frac{\sum x_i}{n}")
    st.latex(
        rf"\bar{{x}} = \frac{{{lista_corta(x)}}}{{{n}}} = \frac{{{suma:.4f}}}{{{n}}} = {media:.4f}\ \text{{{unidad}}}"
    )

    st.markdown("**2. Mediana** (valor central de los datos ordenados)")
    ordenados = sorted(datos)
    st.write("Datos ordenados:", ", ".join(f"{v:g}" for v in ordenados))
    if n % 2 == 1:
        st.write(f"Como n = {n} es impar, la mediana es el dato del centro: **{mediana:g} {unidad}**.")
    else:
        a, b = ordenados[n // 2 - 1], ordenados[n // 2]
        st.write(
            f"Como n = {n} es par, se promedian los dos datos centrales: "
            f"({a:g} + {b:g}) / 2 = **{mediana:g} {unidad}**."
        )

    st.markdown("**3. Moda** (valor que más se repite)")
    if len(modas) == len(set(datos)):
        st.write("Ningún valor se repite, así que no hay moda.")
    else:
        st.write("Valor(es) más frecuente(s): " + ", ".join(f"{m:g}" for m in modas))

    st.markdown("**4. Rango** (diferencia entre el mayor y el menor)")
    st.latex(
        rf"R = x_{{max}} - x_{{min}} = {x.max():g} - {x.min():g} = {rango:.4f}\ \text{{{unidad}}}"
    )

    st.markdown("**5. Desviaciones respecto a la media**")
    tabla = pd.DataFrame(
        {
            "xᵢ": x,
            "xᵢ − media": desviaciones,
            "(xᵢ − media)²": desviaciones ** 2,
        }
    )
    st.dataframe(tabla.round(4), hide_index=True)
    st.write(f"Suma de (xᵢ − media)² = **{suma_cuadrados:.4f}**")

    st.markdown("**6. Desviación estándar muestral** (se divide entre n − 1)")
    st.latex(r"s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}")
    st.latex(
        rf"s = \sqrt{{\frac{{{suma_cuadrados:.4f}}}{{{n - 1}}}}} = \sqrt{{{var_muestral:.4f}}} = {s:.4f}\ \text{{{unidad}}}"
    )

    st.markdown("**7. Desviación estándar poblacional** (se divide entre n)")
    st.latex(r"\sigma = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n}}")
    st.latex(
        rf"\sigma = \sqrt{{\frac{{{suma_cuadrados:.4f}}}{{{n}}}}} = \sqrt{{{var_poblacional:.4f}}} = {sigma:.4f}\ \text{{{unidad}}}"
    )

    # ----- Interpretación -----
    st.subheader("Interpretación")
    st.write(
        f"Las {n} mediciones están centradas alrededor de **{media:.3f} {unidad}**. "
        f"En promedio, cada medición se aleja unos **{s:.3f} {unidad}** de la media "
        f"(desviación muestral), y el rango total de la variación es {rango:.3f} {unidad}."
    )
    st.write(
        f"**¿Cuál desviación usar?** Si las mediciones son una *muestra* de todas las que "
        f"podrían hacerse (lo normal en un laboratorio), se usa la muestral s = {s:.4f} {unidad}, "
        f"que divide entre n − 1 y corrige la tendencia a subestimar la dispersión. "
        f"Solo si tienes *todos* los datos de la población se usa σ = {sigma:.4f} {unidad}. "
        f"Con pocos datos la diferencia es notable; con muchos se vuelve pequeña."
    )
    st.write(
        f"Para reportar el resultado de la medida se suele usar la incertidumbre estándar de la media, "
        f"s/√n = {incert_media:.4f} {unidad}, es decir: **({media:.3f} ± {incert_media:.3f}) {unidad}**."
    )


# ---------------------------------------------------------------
# MÓDULO 2: TABLAS DE FRECUENCIAS Y GRÁFICAS
# ---------------------------------------------------------------
def modulo_tablas_graficas():
    st.header("Tablas de frecuencias y gráficas")
    st.write(
        "Escribe tus mediciones (usa varias, por ejemplo 10 o más, para que el "
        "histograma tenga sentido). Esta sección se actualiza sola al cambiar los datos."
    )

    ejemplo = "2.1, 2.3, 2.2, 2.4, 2.0, 2.2, 2.3, 2.1, 2.5, 2.2, 2.4, 2.3"
    entrada = pedir_datos("tablas", ejemplo, con_boton=False)
    if entrada is None:
        return
    datos, unidad = entrada

    if len(datos) < 2:
        st.warning("Con un solo dato no se puede armar una tabla de frecuencias. Ingresa al menos 2.")
        return

    x = np.array(datos)
    n = len(x)
    minimo, maximo = x.min(), x.max()
    rango = maximo - minimo
    media = x.mean()
    s = x.std(ddof=1)

    if rango == 0:
        st.warning(
            f"Todas las mediciones valen lo mismo ({minimo:g} {unidad}). "
            "No hay variación, así que no se pueden formar intervalos."
        )
        return

    # ----- Número de clases -----
    k_sturges = int(np.ceil(1 + np.log2(n)))
    usar_sturges = st.checkbox(
        "Calcular el número de clases con la regla de Sturges", value=True, key="tablas_sturges"
    )
    if usar_sturges:
        k = k_sturges
    else:
        k = int(
            st.number_input(
                "Número de clases", min_value=2, max_value=30, value=k_sturges, step=1, key="tablas_k"
            )
        )

    amplitud = rango / k
    bordes = np.linspace(minimo, maximo, k + 1)
    f, _ = np.histogram(x, bins=bordes)   # f = frecuencia absoluta de cada clase
    fr = f / n                            # frecuencia relativa
    F = np.cumsum(f)                      # frecuencia acumulada
    marcas = (bordes[:-1] + bordes[1:]) / 2

    # ----- Tabla de frecuencias -----
    st.subheader("Tabla de frecuencias")
    nombres_clase = []
    for i in range(k):
        cierre = "]" if i == k - 1 else ")"   # la última clase incluye su límite superior
        nombres_clase.append(f"[{bordes[i]:.3f}, {bordes[i + 1]:.3f}{cierre}")
    tabla = pd.DataFrame(
        {
            f"Clase ({unidad})": nombres_clase,
            "Marca de clase": marcas.round(4),
            "f": f,
            "fr": fr.round(4),
            "F": F,
            "%": (fr * 100).round(2),
        }
    )
    st.dataframe(tabla, hide_index=True)
    st.write(f"Total de datos: **{int(f.sum())}** (debe coincidir con n = {n}).")

    # ----- Procedimiento -----
    st.subheader("Procedimiento paso a paso")
    st.markdown("**1. Rango de los datos**")
    st.latex(rf"R = x_{{max}} - x_{{min}} = {maximo:g} - {minimo:g} = {rango:.4f}\ \text{{{unidad}}}")

    st.markdown("**2. Número de clases** (regla de Sturges)")
    st.latex(r"k = 1 + \log_2(n)")
    st.latex(
        rf"k = 1 + \log_2({n}) = 1 + {np.log2(n):.3f} = {1 + np.log2(n):.3f} \approx {k_sturges}"
    )
    if not usar_sturges:
        st.write(f"Elegiste usar **{k}** clases en lugar de las {k_sturges} de Sturges.")

    st.markdown("**3. Amplitud de cada clase**")
    st.latex(r"A = \frac{R}{k}")
    st.latex(rf"A = \frac{{{rango:.4f}}}{{{k}}} = {amplitud:.4f}\ \text{{{unidad}}}")

    st.markdown("**4. Límites y marca de clase**")
    st.write(
        f"Se empieza en el valor mínimo ({minimo:g}) y se suma la amplitud ({amplitud:.4f}) "
        "para obtener el límite de cada clase."
    )
    st.latex(r"\text{marca} = \frac{L_{inf} + L_{sup}}{2}")

    st.markdown("**5. Frecuencias**")
    st.write(
        "Se cuenta cuántos datos caen en cada clase (**f**). "
        "La frecuencia relativa es **fr = f / n**, la acumulada **F** suma las f hasta esa clase, "
        "y el porcentaje es fr × 100."
    )
    st.latex(rf"fr_1 = \frac{{f_1}}{{n}} = \frac{{{f[0]}}}{{{n}}} = {fr[0]:.4f}")

    # ----- Gráficas -----
    st.subheader("Histograma")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.hist(x, bins=bordes, edgecolor="white")
    ax1.axvline(media, linestyle="--", color="red", label=f"media = {media:.3f} {unidad}")
    ax1.set_xlabel(f"Valor medido ({unidad})")
    ax1.set_ylabel("Frecuencia (f)")
    ax1.set_title("Histograma de las mediciones")
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Diagrama de dispersión")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    numeros = np.arange(1, n + 1)
    ax2.scatter(numeros, x, zorder=3, label="mediciones")
    ax2.axhline(media, linestyle="--", color="red", label=f"media = {media:.3f} {unidad}")
    ax2.axhspan(media - s, media + s, alpha=0.2, color="orange", label=f"media ± s ({s:.3f} {unidad})")
    ax2.set_xlabel("Número de medición")
    ax2.set_ylabel(f"Valor medido ({unidad})")
    ax2.set_title("Dispersión de las mediciones")
    if n <= 20:
        ax2.set_xticks(numeros)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    # ----- Interpretación -----
    st.subheader("Interpretación")
    i_max = int(np.argmax(f))
    dentro = int(((x >= media - s) & (x <= media + s)).sum())
    st.write(
        f"La clase más frecuente es **{nombres_clase[i_max]} {unidad}**, con {int(f[i_max])} datos "
        f"({fr[i_max] * 100:.1f} % del total). Es el rango de valores donde más se repiten las mediciones."
    )
    st.write(
        f"En el diagrama de dispersión, **{dentro} de {n}** mediciones ({dentro / n * 100:.0f} %) "
        f"quedan dentro de una desviación estándar de la media ({media - s:.3f} a {media + s:.3f} {unidad}). "
        "Mientras más puntos estén cerca de la línea roja, más precisas fueron las mediciones."
    )


# ---------------------------------------------------------------
# MÓDULO 3: PROBABILIDAD Y DISTRIBUCIÓN NORMAL
# ---------------------------------------------------------------
def _probabilidad_clasica():
    st.write("En la **probabilidad clásica** todos los resultados son igual de probables:")
    st.latex(r"P(A) = \frac{\text{casos favorables}}{\text{casos posibles}}")

    c1, c2 = st.columns(2)
    with c1:
        favorables = st.number_input("Casos favorables", min_value=0, value=3, step=1, key="pc_fav")
    with c2:
        posibles = st.number_input("Casos posibles", min_value=0, value=10, step=1, key="pc_pos")

    if posibles == 0:
        st.warning("Los casos posibles deben ser al menos 1.")
        return
    if favorables > posibles:
        st.error("Los casos favorables no pueden ser mayores que los casos posibles.")
        return

    p = favorables / posibles
    st.subheader("Procedimiento y resultado")
    st.latex(rf"P(A) = \frac{{{favorables}}}{{{posibles}}} = {p:.4f} = {p * 100:.2f}\,\%")
    st.latex(rf"P(A^c) = 1 - P(A) = 1 - {p:.4f} = {1 - p:.4f}")
    st.subheader("Interpretación")
    st.write(
        f"El evento A ocurre con probabilidad **{p:.4f}** ({p * 100:.2f} %): "
        f"aproximadamente {p * 100:.0f} de cada 100 veces que se repita el experimento. "
        f"La probabilidad de que NO ocurra (complemento) es {1 - p:.4f}."
    )


def _dos_eventos():
    st.write("Calcula la intersección, la unión, los complementos y la probabilidad condicional.")
    c1, c2 = st.columns(2)
    with c1:
        pA = st.number_input("P(A)", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="de_pa")
    with c2:
        pB = st.number_input("P(B)", min_value=0.0, max_value=1.0, value=0.4, step=0.05, key="de_pb")

    relacion = st.radio(
        "¿Cómo se relacionan A y B?",
        ["Independientes", "Mutuamente excluyentes", "Conozco P(A ∩ B)"],
        horizontal=True,
        key="de_relacion",
    )

    if relacion == "Independientes":
        pAB = pA * pB
        formula_inter = rf"P(A \cap B) = P(A)\,P(B) = {pA:.4f} \times {pB:.4f} = {pAB:.4f}"
    elif relacion == "Mutuamente excluyentes":
        if pA + pB > 1:
            st.error(
                "Si A y B son mutuamente excluyentes, P(A) + P(B) no puede pasar de 1. "
                "Revisa los valores."
            )
            return
        pAB = 0.0
        formula_inter = r"P(A \cap B) = 0 \quad \text{(no pueden ocurrir a la vez)}"
    else:
        pAB = st.number_input(
            "P(A ∩ B)", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="de_pab"
        )
        if pAB > min(pA, pB):
            st.error("P(A ∩ B) no puede ser mayor que P(A) ni que P(B).")
            return
        if pAB < pA + pB - 1:
            st.error("Con esos valores la unión pasaría de 1. P(A ∩ B) debe ser al menos P(A) + P(B) − 1.")
            return
        formula_inter = rf"P(A \cap B) = {pAB:.4f}"

    pU = pA + pB - pAB

    st.subheader("Procedimiento y resultado")
    st.markdown("**1. Intersección** (ocurren A y B a la vez)")
    st.latex(formula_inter)
    st.markdown("**2. Unión** (ocurre A, B o ambos)")
    st.latex(r"P(A \cup B) = P(A) + P(B) - P(A \cap B)")
    st.latex(rf"P(A \cup B) = {pA:.4f} + {pB:.4f} - {pAB:.4f} = {pU:.4f}")
    st.markdown("**3. Complementos**")
    st.latex(rf"P(A^c) = 1 - {pA:.4f} = {1 - pA:.4f} \qquad P(B^c) = 1 - {pB:.4f} = {1 - pB:.4f}")
    st.markdown("**4. Probabilidad condicional**")
    if pB > 0:
        pA_dado_B = pAB / pB
        st.latex(r"P(A \mid B) = \frac{P(A \cap B)}{P(B)}")
        st.latex(rf"P(A \mid B) = \frac{{{pAB:.4f}}}{{{pB:.4f}}} = {pA_dado_B:.4f}")
    else:
        st.write("P(A | B) no está definida porque P(B) = 0.")

    st.subheader("Interpretación")
    st.write(
        f"La probabilidad de que ocurra **A o B** es {pU:.4f} ({pU * 100:.2f} %), "
        f"y de que ocurran **los dos** es {pAB:.4f} ({pAB * 100:.2f} %)."
    )
    if abs(pAB - pA * pB) < 1e-9:
        st.write("Como P(A ∩ B) = P(A)·P(B), los eventos son **independientes**.")
    elif pAB == 0:
        st.write("Como P(A ∩ B) = 0, los eventos son **mutuamente excluyentes**.")
    else:
        st.write("Como P(A ∩ B) ≠ P(A)·P(B), los eventos **no son independientes**: saber que ocurrió uno cambia la probabilidad del otro.")


def _distribucion_normal():
    st.write(
        "Si las mediciones siguen una distribución normal con media μ y desviación σ, "
        "se puede calcular la probabilidad de que una medición caiga en cierto rango."
    )

    fuente = st.radio(
        "¿De dónde salen μ y σ?",
        ["Los escribo yo", "De mis mediciones"],
        horizontal=True,
        key="norm_fuente",
    )
    if fuente == "Los escribo yo":
        c1, c2, c3 = st.columns(3)
        with c1:
            mu = st.number_input("Media (μ)", value=2.2, format="%.4f", key="norm_mu")
        with c2:
            sigma = st.number_input("Desviación estándar (σ)", value=0.1581, format="%.4f", key="norm_sigma")
        with c3:
            unidad = st.text_input("Unidad", value="A", key="norm_unidad")
    else:
        entrada = pedir_datos("normal", "2.1, 2.3, 2.2, 2.4, 2.0", con_boton=False)
        if entrada is None:
            return
        datos, unidad = entrada
        if len(datos) < 2:
            st.warning("Se necesitan al menos 2 mediciones para calcular la desviación muestral.")
            return
        mu = float(np.mean(datos))
        sigma = float(np.std(datos, ddof=1))
        st.info(f"Se usará μ = {mu:.4f} {unidad} (la media) y σ = s = {sigma:.4f} {unidad} (desviación muestral).")

    if sigma <= 0:
        st.error("La desviación estándar debe ser mayor que 0.")
        return

    tipo = st.radio(
        "Probabilidad que quieres calcular",
        ["P(X < a)", "P(X > a)", "P(a < X < b)"],
        horizontal=True,
        key="norm_tipo",
    )
    entre = tipo == "P(a < X < b)"
    c1, c2 = st.columns(2)
    with c1:
        a = st.number_input("Valor a", value=2.1, format="%.4f", key="norm_a")
    with c2:
        b = st.number_input("Valor b", value=2.4, format="%.4f", key="norm_b", disabled=not entre)

    if entre and a >= b:
        st.error("Para P(a < X < b) el valor a debe ser menor que b.")
        return

    # ----- Cálculos -----
    za = (a - mu) / sigma
    zb = (b - mu) / sigma
    if tipo == "P(X < a)":
        p = cdf_normal(za)
    elif tipo == "P(X > a)":
        p = 1 - cdf_normal(za)
    else:
        p = cdf_normal(zb) - cdf_normal(za)

    # ----- Procedimiento -----
    st.subheader("Procedimiento paso a paso")
    st.markdown("**1. Estandarizar** (convertir a puntaje z)")
    st.latex(r"z = \frac{x - \mu}{\sigma}")
    st.latex(rf"z_a = \frac{{{a:g} - {mu:.4f}}}{{{sigma:.4f}}} = {za:.4f}")
    if entre:
        st.latex(rf"z_b = \frac{{{b:g} - {mu:.4f}}}{{{sigma:.4f}}} = {zb:.4f}")

    st.markdown("**2. Buscar la probabilidad acumulada** Φ(z) (es lo que da la tabla de la normal estándar)")
    st.latex(rf"\Phi({za:.4f}) = {cdf_normal(za):.4f}")
    if entre:
        st.latex(rf"\Phi({zb:.4f}) = {cdf_normal(zb):.4f}")
    st.caption(
        "Aquí se usa z con 4 decimales. Si buscas en una tabla Z impresa, redondea z a 2 decimales: "
        f"Φ({za:.2f}) = {cdf_normal(round(za, 2)):.4f}"
        + (f" y Φ({zb:.2f}) = {cdf_normal(round(zb, 2)):.4f}" if entre else "")
        + ". La pequeña diferencia viene solo del redondeo."
    )

    st.markdown("**3. Calcular la probabilidad pedida**")
    if tipo == "P(X < a)":
        st.latex(rf"P(X < {a:g}) = \Phi({za:.4f}) = {p:.4f}")
    elif tipo == "P(X > a)":
        st.latex(rf"P(X > {a:g}) = 1 - \Phi({za:.4f}) = 1 - {cdf_normal(za):.4f} = {p:.4f}")
    else:
        st.latex(
            rf"P({a:g} < X < {b:g}) = \Phi({zb:.4f}) - \Phi({za:.4f}) = "
            rf"{cdf_normal(zb):.4f} - {cdf_normal(za):.4f} = {p:.4f}"
        )

    st.metric("Probabilidad", f"{p:.4f}  ({p * 100:.2f} %)")

    # ----- Gráfica -----
    lo = min(mu - 4 * sigma, a - sigma)
    hi = max(mu + 4 * sigma, (b if entre else a) + sigma)
    xs = np.linspace(lo, hi, 500)
    ys = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    if tipo == "P(X < a)":
        zona = xs <= a
    elif tipo == "P(X > a)":
        zona = xs >= a
    else:
        zona = (xs >= a) & (xs <= b)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys)
    ax.fill_between(xs, ys, where=zona, alpha=0.4, label=f"{tipo} = {p:.4f}")
    ax.axvline(mu, linestyle="--", color="red", label=f"μ = {mu:.3f} {unidad}")
    ax.set_xlabel(f"x ({unidad})")
    ax.set_ylabel("Densidad de probabilidad")
    ax.set_title("Distribución normal")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # ----- Interpretación -----
    st.subheader("Interpretación")
    if tipo == "P(X < a)":
        frase = f"sea menor que {a:g} {unidad}"
    elif tipo == "P(X > a)":
        frase = f"sea mayor que {a:g} {unidad}"
    else:
        frase = f"esté entre {a:g} y {b:g} {unidad}"
    st.write(
        f"Hay una probabilidad de **{p:.4f}** ({p * 100:.2f} %) de que una medición {frase}. "
        f"Dicho de otro modo, en unas {p * 100:.0f} de cada 100 mediciones ocurriría esto."
    )
    r1, r2, r3 = (math.erf(k / math.sqrt(2)) * 100 for k in (1, 2, 3))
    st.write(
        f"Regla empírica: en una normal, el {r1:.1f} % de los datos queda a ±1σ de la media "
        f"({mu - sigma:.3f} a {mu + sigma:.3f} {unidad}), el {r2:.1f} % a ±2σ y el {r3:.1f} % a ±3σ."
    )
    st.caption(
        "Este resultado vale si las mediciones realmente se distribuyen de forma normal "
        "(el histograma del módulo anterior debería parecerse a una campana)."
    )


def modulo_probabilidad():
    st.header("Probabilidad y distribución normal")
    tema = st.radio(
        "¿Qué quieres resolver?",
        ["Probabilidad clásica", "Dos eventos", "Distribución normal"],
        horizontal=True,
        key="prob_tema",
    )
    if tema == "Probabilidad clásica":
        _probabilidad_clasica()
    elif tema == "Dos eventos":
        _dos_eventos()
    else:
        _distribucion_normal()


# ---------------------------------------------------------------
# MÓDULO 4: CORRELACIÓN Y REGRESIÓN LINEAL
# ---------------------------------------------------------------
def fuerza_correlacion(r):
    """Describe con palabras qué tan fuerte es la correlación según |r|."""
    a = abs(r)
    if a >= 0.9:
        return "muy fuerte"
    if a >= 0.7:
        return "fuerte"
    if a >= 0.4:
        return "moderada"
    if a >= 0.2:
        return "débil"
    return "muy débil o nula"


def modulo_regresion():
    st.header("Correlación y regresión lineal")
    st.write(
        "Escribe dos series de mediciones de la misma longitud: **x** (la variable que "
        "controlas o varías) e **y** (la que mides). Se actualiza sola al cambiar los datos."
    )

    c1, c2 = st.columns(2)
    with c1:
        nombre_x = st.text_input("Nombre de x", value="Corriente I", key="reg_nx")
        unidad_x = st.text_input("Unidad de x", value="A", key="reg_ux")
        texto_x = st.text_area("Valores de x", value="0.2, 0.4, 0.6, 0.8, 1.0", height=100, key="reg_x")
    with c2:
        nombre_y = st.text_input("Nombre de y", value="Voltaje V", key="reg_ny")
        unidad_y = st.text_input("Unidad de y", value="V", key="reg_uy")
        texto_y = st.text_area("Valores de y", value="1.1, 2.0, 3.1, 3.9, 5.1", height=100, key="reg_y")

    # ----- Validación -----
    datos_x, errores_x = leer_datos(texto_x)
    datos_y, errores_y = leer_datos(texto_y)
    if errores_x or errores_y:
        partes = []
        if errores_x:
            partes.append("en x: " + ", ".join(f"'{e}'" for e in errores_x))
        if errores_y:
            partes.append("en y: " + ", ".join(f"'{e}'" for e in errores_y))
        st.error("Hay valores que no son números (" + "; ".join(partes) + "). Corrígelos.")
        return
    if len(datos_x) == 0 or len(datos_y) == 0:
        st.warning("Faltan datos: escribe valores tanto en x como en y.")
        return
    if len(datos_x) != len(datos_y):
        st.error(
            f"x tiene {len(datos_x)} valores e y tiene {len(datos_y)}. "
            "Deben tener la misma cantidad (un valor de y por cada x)."
        )
        return
    if len(datos_x) < 3:
        st.warning("Se necesitan al menos 3 pares de datos para ajustar una recta con incertidumbre.")
        return

    x = np.array(datos_x)
    y = np.array(datos_y)
    n = len(x)
    if x.max() == x.min():
        st.error("Todos los valores de x son iguales: no se puede ajustar una recta.")
        return
    if y.max() == y.min():
        st.warning(
            "Todos los valores de y son iguales: la recta es horizontal (pendiente 0) "
            "y el coeficiente de correlación r no está definido."
        )
        return

    # ----- Cálculos -----
    x_media = x.mean()
    y_media = y.mean()
    dx = x - x_media
    dy = y - y_media
    Sxx = float((dx ** 2).sum())
    Syy = float((dy ** 2).sum())
    Sxy = float((dx * dy).sum())
    r = Sxy / math.sqrt(Sxx * Syy)
    m = Sxy / Sxx                    # pendiente
    b = y_media - m * x_media        # intercepto
    r2 = r ** 2
    y_ajustada = m * x + b
    sse = float(((y - y_ajustada) ** 2).sum())
    s_e = math.sqrt(sse / (n - 2))   # desviación de los residuos
    s_m = s_e / math.sqrt(Sxx)       # incertidumbre de la pendiente
    s_b = s_e * math.sqrt(1 / n + x_media ** 2 / Sxx)  # incertidumbre del intercepto
    signo = "+" if b >= 0 else "−"
    ecuacion = f"y = {m:.4f}·x {signo} {abs(b):.4f}"

    # ----- Resultados -----
    st.subheader("Resultados")
    r1, r2c, r3 = st.columns(3)
    r1.metric("Correlación r", f"{r:.4f}")
    r2c.metric("R²", f"{r2:.4f}")
    r3.metric("Pendiente m", f"{m:.4f} {unidad_y}/{unidad_x}")
    st.write(f"**Ecuación de la recta:** {ecuacion}")

    # ----- Procedimiento -----
    st.subheader("Procedimiento paso a paso")
    st.markdown("**1. Medias y desviaciones**")
    st.latex(rf"\bar{{x}} = {x_media:.4f} \qquad \bar{{y}} = {y_media:.4f}")
    tabla = pd.DataFrame(
        {
            "xᵢ": x,
            "yᵢ": y,
            "xᵢ − x̄": dx,
            "yᵢ − ȳ": dy,
            "(xᵢ − x̄)(yᵢ − ȳ)": dx * dy,
            "(xᵢ − x̄)²": dx ** 2,
            "(yᵢ − ȳ)²": dy ** 2,
        }
    )
    st.dataframe(tabla.round(4), hide_index=True)

    st.markdown("**2. Sumas de cuadrados y productos**")
    st.latex(
        rf"S_{{xx}} = \sum (x_i - \bar{{x}})^2 = {Sxx:.4f} \qquad "
        rf"S_{{yy}} = \sum (y_i - \bar{{y}})^2 = {Syy:.4f} \qquad "
        rf"S_{{xy}} = \sum (x_i - \bar{{x}})(y_i - \bar{{y}}) = {Sxy:.4f}"
    )

    st.markdown("**3. Coeficiente de correlación lineal**")
    st.latex(r"r = \frac{S_{xy}}{\sqrt{S_{xx}\,S_{yy}}}")
    st.latex(rf"r = \frac{{{Sxy:.4f}}}{{\sqrt{{({Sxx:.4f})({Syy:.4f})}}}} = {r:.4f}")

    st.markdown("**4. Recta de mínimos cuadrados** y = m·x + b")
    st.latex(r"m = \frac{S_{xy}}{S_{xx}} \qquad b = \bar{y} - m\,\bar{x}")
    st.latex(rf"m = \frac{{{Sxy:.4f}}}{{{Sxx:.4f}}} = {m:.4f}")
    st.latex(rf"b = {y_media:.4f} - ({m:.4f})({x_media:.4f}) = {b:.4f}")

    st.markdown("**5. Coeficiente de determinación**")
    st.latex(rf"R^2 = r^2 = ({r:.4f})^2 = {r2:.4f}")

    st.markdown("**6. Incertidumbre de la pendiente y del intercepto**")
    st.latex(r"s_e = \sqrt{\frac{\sum (y_i - \hat{y}_i)^2}{n-2}} \qquad s_m = \frac{s_e}{\sqrt{S_{xx}}}")
    st.latex(
        rf"s_e = \sqrt{{\frac{{{sse:.4f}}}{{{n - 2}}}}} = {s_e:.4f} \qquad "
        rf"s_m = \frac{{{s_e:.4f}}}{{\sqrt{{{Sxx:.4f}}}}} = {s_m:.4f}"
    )
    st.latex(rf"s_b = s_e\sqrt{{\frac{{1}}{{n}} + \frac{{\bar{{x}}^2}}{{S_{{xx}}}}}} = {s_b:.4f}")

    # ----- Gráfica -----
    st.subheader("Gráfica")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, zorder=3, label="mediciones")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m * xs + b, color="red", label=f"{ecuacion}  (r = {r:.3f})")
    ax.set_xlabel(f"{nombre_x} ({unidad_x})")
    ax.set_ylabel(f"{nombre_y} ({unidad_y})")
    ax.set_title("Regresión lineal")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # ----- Predicción -----
    st.subheader("Predicción")
    x_nuevo = st.number_input(
        f"Valor de x para predecir y ({unidad_x})", value=0.5, format="%.4f", key="reg_pred"
    )
    y_nuevo = m * x_nuevo + b
    st.latex(rf"\hat{{y}} = ({m:.4f})({x_nuevo:g}) + ({b:.4f}) = {y_nuevo:.4f}\ \text{{{unidad_y}}}")
    if x_nuevo < x.min() or x_nuevo > x.max():
        st.warning(
            f"Ese valor está fuera del rango medido ({x.min():g} a {x.max():g}). "
            "Es una extrapolación y puede no ser confiable."
        )

    # ----- Interpretación -----
    st.subheader("Interpretación")
    direccion = "positiva (si x aumenta, y también)" if r > 0 else "negativa (si x aumenta, y disminuye)"
    st.write(
        f"El coeficiente r = **{r:.4f}** indica una correlación lineal **{fuerza_correlacion(r)}** "
        f"y {direccion}. Además, R² = {r2:.4f}: el **{r2 * 100:.1f} %** de la variación de "
        f"{nombre_y} se explica por su relación lineal con {nombre_x}."
    )
    st.write(
        f"La pendiente es **m = ({m:.4f} ± {s_m:.4f}) {unidad_y}/{unidad_x}**: por cada {unidad_x} "
        f"que aumenta {nombre_x}, {nombre_y} cambia en promedio {m:.4f} {unidad_y}. "
        f"El intercepto es b = ({b:.4f} ± {s_b:.4f}) {unidad_y}."
    )
    if unidad_y.strip() == "V" and unidad_x.strip() == "A":
        st.write(
            f"**En Física:** al graficar voltaje contra corriente, la pendiente es la resistencia "
            f"(ley de Ohm, V = I·R). Aquí **R = ({m:.3f} ± {s_m:.3f}) Ω**, y el intercepto debería "
            f"estar cerca de 0."
        )
    st.caption(
        "Recuerda: r mide solo relación **lineal**, y una correlación alta no demuestra por sí sola "
        "que x cause a y."
    )


# ---------------------------------------------------------------
# MÓDULO 5: EJERCICIOS RESUELTOS Y PRÁCTICA CON RETROALIMENTACIÓN
# ---------------------------------------------------------------
# Los "resolvedores" no dibujan nada: solo calculan y devuelven
#   - "pasos": lista de (tipo, contenido) con tipo "texto", "latex" o "tabla"
#   - "valores": diccionario con los resultados numéricos
# Así sirven tanto para los ejercicios resueltos como para corregir las prácticas.
def decimales_de(valores, tope=4):
    """Cuenta cuántos decimales usan los datos (para escribir bien la sustitución)."""
    mayor = 0
    for v in valores:
        texto = f"{v:.{tope}f}".rstrip("0")
        if "." in texto:
            mayor = max(mayor, len(texto.split(".")[1]))
    return mayor


def mostrar_pasos(pasos):
    for tipo, contenido in pasos:
        if tipo == "latex":
            st.latex(contenido)
        elif tipo == "tabla":
            st.dataframe(contenido, hide_index=True)
        else:
            st.markdown(contenido)


def resolver_descriptiva(datos, unidad, pedir):
    """'pedir' es una lista con: media, mediana, moda, rango, s, sigma."""
    x = np.array(datos, dtype=float)
    n = len(x)
    d = decimales_de(datos)
    media = float(x.mean())
    suma = float(x.sum())
    suma_cuad = float(((x - media) ** 2).sum())
    var_s = suma_cuad / (n - 1)
    var_p = suma_cuad / n
    orden = sorted(float(t) for t in datos)
    if n % 2 == 1:
        mediana = orden[n // 2]
    else:
        mediana = (orden[n // 2 - 1] + orden[n // 2]) / 2
    modas = statistics.multimode(orden)
    sin_moda = len(modas) == len(set(orden))
    rango = float(x.max() - x.min())

    valores = {
        "media": media, "mediana": mediana, "rango": rango,
        "moda": None if sin_moda else modas[0],
        "var": var_s, "s": math.sqrt(var_s), "sigma": math.sqrt(var_p),
    }
    suma_t = f"{suma:.{d + 1}f}"
    ordenados_t = ", ".join(f"{t:g}" for t in orden)
    pasos = []

    if "media" in pedir:
        pasos += [
            ("texto", "**Media aritmética**"),
            ("latex", r"\bar{x} = \frac{\sum x_i}{n}"),
            ("latex", rf"\bar{{x}} = \frac{{{lista_corta(x, d)}}}{{{n}}} = \frac{{{suma_t}}}{{{n}}} = {media:.4f}\ \text{{{unidad}}}"),
        ]
    if "mediana" in pedir:
        pasos.append(("texto", f"**Mediana.** Datos ordenados: {ordenados_t}."))
        if n % 2 == 1:
            pasos.append(("texto", f"Como n = {n} es impar, la mediana es el dato central: **{mediana:g} {unidad}**."))
        else:
            a_, b_ = orden[n // 2 - 1], orden[n // 2]
            pasos.append(("texto", f"Como n = {n} es par, se promedian los dos centrales: ({a_:g} + {b_:g}) / 2 = **{mediana:g} {unidad}**."))
    if "moda" in pedir:
        if sin_moda:
            pasos.append(("texto", "**Moda.** Ningún valor se repite, así que no hay moda."))
        else:
            pasos.append(("texto", "**Moda** (el valor que más se repite): " + ", ".join(f"{m:g}" for m in modas) + f" {unidad}."))
    if "rango" in pedir:
        pasos += [
            ("texto", "**Rango**"),
            ("latex", rf"R = x_{{max}} - x_{{min}} = {max(orden):g} - {min(orden):g} = {rango:.4f}\ \text{{{unidad}}}"),
        ]
    if "s" in pedir or "sigma" in pedir:
        if "media" not in pedir:
            pasos.append(("latex", rf"\bar{{x}} = \frac{{{suma_t}}}{{{n}}} = {media:.4f}\ \text{{{unidad}}}"))
        pasos.append(("texto", f"**Suma de las desviaciones al cuadrado:** Σ(xᵢ − x̄)² = {suma_cuad:.4f}"))
    if "s" in pedir:
        pasos += [
            ("texto", "**Desviación estándar muestral** (se divide entre n − 1)"),
            ("latex", r"s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}"),
            ("latex", rf"s = \sqrt{{\frac{{{suma_cuad:.4f}}}{{{n - 1}}}}} = \sqrt{{{var_s:.4f}}} = {valores['s']:.4f}\ \text{{{unidad}}}"),
        ]
    if "sigma" in pedir:
        pasos += [
            ("texto", "**Desviación estándar poblacional** (se divide entre n)"),
            ("latex", r"\sigma = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n}}"),
            ("latex", rf"\sigma = \sqrt{{\frac{{{suma_cuad:.4f}}}{{{n}}}}} = \sqrt{{{var_p:.4f}}} = {valores['sigma']:.4f}\ \text{{{unidad}}}"),
        ]
    return {"pasos": pasos, "valores": valores}


def resolver_sturges(datos, unidad):
    x = np.array(datos, dtype=float)
    n = len(x)
    rango = float(x.max() - x.min())
    k = int(np.ceil(1 + np.log2(n)))
    amplitud = rango / k
    bordes = np.linspace(x.min(), x.max(), k + 1)
    f, _ = np.histogram(x, bins=bordes)
    nombres = [
        f"[{bordes[i]:.3f}, {bordes[i + 1]:.3f}{']' if i == k - 1 else ')'}" for i in range(k)
    ]
    tabla = pd.DataFrame(
        {f"Clase ({unidad})": nombres, "f": f, "fr": (f / n).round(4), "F": np.cumsum(f)}
    )
    pasos = [
        ("latex", rf"R = {x.max():g} - {x.min():g} = {rango:.4f}\ \text{{{unidad}}}"),
        ("latex", rf"k = 1 + \log_2({n}) = {1 + np.log2(n):.3f} \approx {k}"),
        ("latex", rf"A = \frac{{R}}{{k}} = \frac{{{rango:.4f}}}{{{k}}} = {amplitud:.4f}\ \text{{{unidad}}}"),
        ("tabla", tabla),
    ]
    return {"pasos": pasos, "valores": {"k": k, "A": amplitud, "f": f.tolist()}}


def resolver_clasica(favorables, posibles):
    p = favorables / posibles
    pasos = [
        ("latex", r"P(A) = \frac{\text{casos favorables}}{\text{casos posibles}}"),
        ("latex", rf"P(A) = \frac{{{favorables}}}{{{posibles}}} = {p:.4f}"),
    ]
    return {"pasos": pasos, "valores": {"p": p, "complemento": 1 - p}}


def resolver_independientes(pA, pB):
    pAB = pA * pB
    pU = pA + pB - pAB
    pasos = [
        ("latex", rf"P(A \cap B) = P(A)\,P(B) = {pA:g} \times {pB:g} = {pAB:.4f}"),
        ("latex", rf"P(A \cup B) = P(A) + P(B) - P(A \cap B) = {pA:g} + {pB:g} - {pAB:.4f} = {pU:.4f}"),
    ]
    return {"pasos": pasos, "valores": {"interseccion": pAB, "union": pU}}


def resolver_normal(mu, sigma, tipo, a, b=None):
    """tipo: '<' (P(X<a)), '>' (P(X>a)) o 'entre' (P(a<X<b))."""
    za = (a - mu) / sigma
    pasos = [
        ("latex", r"z = \frac{x - \mu}{\sigma}"),
        ("latex", rf"z_a = \frac{{{a:g} - {mu:g}}}{{{sigma:g}}} = {za:.4f}"),
    ]
    if tipo == "<":
        p = cdf_normal(za)
        pasos.append(("latex", rf"P(X < {a:g}) = \Phi({za:.4f}) = {p:.4f}"))
    elif tipo == ">":
        p = 1 - cdf_normal(za)
        pasos.append(("latex", rf"P(X > {a:g}) = 1 - \Phi({za:.4f}) = 1 - {cdf_normal(za):.4f} = {p:.4f}"))
    else:
        zb = (b - mu) / sigma
        p = cdf_normal(zb) - cdf_normal(za)
        pasos.append(("latex", rf"z_b = \frac{{{b:g} - {mu:g}}}{{{sigma:g}}} = {zb:.4f}"))
        pasos.append((
            "latex",
            rf"P({a:g} < X < {b:g}) = \Phi({zb:.4f}) - \Phi({za:.4f}) = {cdf_normal(zb):.4f} - {cdf_normal(za):.4f} = {p:.4f}",
        ))
    return {"pasos": pasos, "valores": {"p": p, "z": za}}


def resolver_regresion(xs, ys, unidad_x, unidad_y):
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    Sxx, Syy, Sxy = float((dx ** 2).sum()), float((dy ** 2).sum()), float((dx * dy).sum())
    r = Sxy / math.sqrt(Sxx * Syy)
    m = Sxy / Sxx
    b = ym - m * xm
    sse = float(((y - (m * x + b)) ** 2).sum())
    s_m = math.sqrt(sse / (n - 2)) / math.sqrt(Sxx)
    pasos = [
        ("latex", rf"\bar{{x}} = {xm:.4f} \qquad \bar{{y}} = {ym:.4f}"),
        ("latex", rf"S_{{xx}} = {Sxx:.4f} \qquad S_{{yy}} = {Syy:.4f} \qquad S_{{xy}} = {Sxy:.4f}"),
        ("latex", rf"r = \frac{{S_{{xy}}}}{{\sqrt{{S_{{xx}}S_{{yy}}}}}} = \frac{{{Sxy:.4f}}}{{\sqrt{{({Sxx:.4f})({Syy:.4f})}}}} = {r:.4f}"),
        ("latex", rf"m = \frac{{S_{{xy}}}}{{S_{{xx}}}} = \frac{{{Sxy:.4f}}}{{{Sxx:.4f}}} = {m:.4f}\ \text{{{unidad_y}/{unidad_x}}}"),
        ("latex", rf"b = \bar{{y}} - m\,\bar{{x}} = {ym:.4f} - ({m:.4f})({xm:.4f}) = {b:.4f}\ \text{{{unidad_y}}}"),
    ]
    return {"pasos": pasos, "valores": {"r": r, "m": m, "b": b, "s_m": s_m, "R2": r * r}}


def ejercicios_resueltos():
    """Lista de los 10 ejercicios de ejemplo. Cada uno calcula su propia solución."""
    corrientes = [2.1, 2.3, 2.2, 2.4, 2.0]
    caidas = [0.52, 0.49, 0.51, 0.50, 0.53, 0.49]
    pendulo = [98.2, 98.5, 98.1, 98.4, 98.3, 98.6, 98.2]
    voltajes = [4.8, 5.1, 5.0, 4.9, 5.2, 5.0, 4.7, 5.1, 5.3, 4.9, 5.0, 5.2]
    corr_ohm = [0.10, 0.20, 0.30, 0.40, 0.50]
    volt_ohm = [0.52, 0.99, 1.55, 2.02, 2.48]
    tiempos = [1, 2, 3, 4, 5]
    distancias = [2.1, 3.9, 6.2, 7.8, 10.1]

    def lista(v):
        return ", ".join(f"{t:g}" for t in v)

    return [
        {
            "titulo": "Corriente en un circuito (ejemplo de prueba)",
            "enunciado": f"Con mediciones de corriente (A): {lista(corrientes)}. Calcula la media y la desviación estándar muestral (y la poblacional para comparar), e interpreta.",
            "resolver": lambda: resolver_descriptiva(corrientes, "A", ["media", "s", "sigma"]),
            "interp": lambda v: (
                f"Las corrientes se concentran alrededor de {v['media']:.2f} A y cada medición se aleja en promedio "
                f"unos {v['s']:.3f} A de la media. Se usa s (muestral) porque son una muestra de las mediciones "
                f"posibles; σ = {v['sigma']:.3f} A solo aplicaría si fueran toda la población."
            ),
        },
        {
            "titulo": "Tiempo de caída de un objeto",
            "enunciado": f"Se midió el tiempo de caída (s) seis veces: {lista(caidas)}. Halla la media, la mediana y la moda.",
            "resolver": lambda: resolver_descriptiva(caidas, "s", ["media", "mediana", "moda"]),
            "interp": lambda v: (
                f"La media ({v['media']:.4f} s) y la mediana ({v['mediana']:.3f} s) son casi iguales, así que las mediciones "
                f"están repartidas de forma bastante simétrica; el valor más repetido es {v['moda']:g} s."
            ),
        },
        {
            "titulo": "Longitud de un péndulo",
            "enunciado": f"Se midió la longitud de un péndulo (cm): {lista(pendulo)}. Calcula el rango y la desviación estándar muestral.",
            "resolver": lambda: resolver_descriptiva(pendulo, "cm", ["rango", "s"]),
            "interp": lambda v: (
                f"Las mediciones varían en un rango de {v['rango']:.1f} cm y la desviación típica es {v['s']:.3f} cm: "
                "son bastante consistentes entre sí."
            ),
        },
        {
            "titulo": "Tabla de frecuencias de voltajes",
            "enunciado": f"Se midió un voltaje (V) doce veces: {lista(voltajes)}. Construye la tabla de frecuencias con la regla de Sturges.",
            "resolver": lambda: resolver_sturges(voltajes, "V"),
            "interp": lambda v: (
                f"Con k = {v['k']} clases de amplitud {v['A']:.3f} V, la clase con más datos tiene {max(v['f'])} mediciones; "
                "ahí se concentra el valor típico del voltaje."
            ),
        },
        {
            "titulo": "Probabilidad clásica: resistores defectuosos",
            "enunciado": "En un lote de 20 resistores hay 4 defectuosos. Si se elige uno al azar, ¿cuál es la probabilidad de que sea defectuoso?",
            "resolver": lambda: resolver_clasica(4, 20),
            "interp": lambda v: (
                f"Hay {v['p'] * 100:.0f} % de probabilidad de sacar uno defectuoso; la probabilidad de sacar uno bueno es {v['complemento']:.2f}."
            ),
        },
        {
            "titulo": "Dos sensores independientes",
            "enunciado": "Dos sensores fallan de forma independiente. P(falla A) = 0.3 y P(falla B) = 0.5. ¿Cuál es la probabilidad de que falle al menos uno?",
            "resolver": lambda: resolver_independientes(0.3, 0.5),
            "interp": lambda v: (
                f"La probabilidad de que fallen ambos a la vez es {v['interseccion']:.2f}; de que falle al menos uno, {v['union']:.2f} ({v['union'] * 100:.0f} %)."
            ),
        },
        {
            "titulo": "Normal: corriente mayor que un límite",
            "enunciado": "Las mediciones de corriente siguen una normal con μ = 2.2 A y σ = 0.16 A. Calcula P(X > 2.4).",
            "resolver": lambda: resolver_normal(2.2, 0.16, ">", 2.4),
            "interp": lambda v: (
                f"Solo alrededor del {v['p'] * 100:.1f} % de las mediciones superaría 2.4 A; está a {v['z']:.2f} desviaciones sobre la media."
            ),
        },
        {
            "titulo": "Normal: resistencia dentro de una tolerancia",
            "enunciado": "La resistencia de un lote es normal con μ = 100 Ω y σ = 5 Ω. ¿Qué probabilidad hay de que una resistencia esté entre 95 Ω y 110 Ω?",
            "resolver": lambda: resolver_normal(100, 5, "entre", 95, 110),
            "interp": lambda v: (
                f"Aproximadamente el {v['p'] * 100:.1f} % de las resistencias queda entre 95 Ω y 110 Ω."
            ),
        },
        {
            "titulo": "Ley de Ohm: resistencia con regresión",
            "enunciado": f"Se midió el voltaje V (V) para distintas corrientes I (A). I: {lista(corr_ohm)}; V: {lista(volt_ohm)}. Ajusta V = m·I + b e interpreta la pendiente.",
            "resolver": lambda: resolver_regresion(corr_ohm, volt_ohm, "A", "V"),
            "interp": lambda v: (
                f"Con r = {v['r']:.4f} la relación es lineal muy fuerte. La pendiente es la resistencia: "
                f"R = ({v['m']:.3f} ± {v['s_m']:.3f}) Ω, y el intercepto b = {v['b']:.3f} V queda cerca de 0, como pide V = I·R."
            ),
        },
        {
            "titulo": "Movimiento uniforme: velocidad con regresión",
            "enunciado": f"Un carrito recorre distancias d (m) en tiempos t (s). t: {lista(tiempos)}; d: {lista(distancias)}. Ajusta d = m·t + b y halla la velocidad.",
            "resolver": lambda: resolver_regresion(tiempos, distancias, "s", "m"),
            "interp": lambda v: (
                f"La pendiente es la velocidad: v = ({v['m']:.3f} ± {v['s_m']:.3f}) m/s. Con r = {v['r']:.4f} el movimiento es muy "
                "cercano a uniforme."
            ),
        },
    ]


def seccion_ejercicios_resueltos():
    st.write(
        "Aquí hay **10 ejercicios resueltos** paso a paso. Abre cada uno para ver las fórmulas, "
        "la sustitución de valores, el resultado y la interpretación."
    )
    for i, ej in enumerate(ejercicios_resueltos(), start=1):
        with st.expander(f"Ejercicio {i}: {ej['titulo']}"):
            st.write(ej["enunciado"])
            res = ej["resolver"]()
            mostrar_pasos(res["pasos"])
            st.success(ej["interp"](res["valores"]))


# ----- Generador de prácticas -----
TEMAS_PRACTICA = [
    "Estadística descriptiva",
    "Probabilidad clásica",
    "Distribución normal",
    "Regresión lineal",
]


def generar_problema(tema):
    """Crea un ejercicio aleatorio. Devuelve un diccionario con el enunciado,
    la respuesta correcta, la tolerancia, la solución y pistas para errores típicos
    (cada pista es (valor_equivocado, tolerancia, mensaje))."""
    if tema == "Estadística descriptiva":
        unidad, nombre = random.choice(
            [("A", "corriente"), ("V", "voltaje"), ("s", "tiempo de caída"), ("cm", "longitud")]
        )
        base = round(random.uniform(1, 12), 1)
        while True:
            n = random.randint(5, 7)
            datos = [round(base + random.uniform(-0.4, 0.4), 2) for _ in range(n)]
            if len(set(datos)) > 1:
                break
        pedir = random.choice(["media", "mediana", "rango", "s"])
        res = resolver_descriptiva(datos, unidad, [pedir])
        v = res["valores"]
        valor = v[pedir]
        if pedir == "s":
            tol = max(0.0015, 0.002 * abs(valor))   # s depende del redondeo intermedio
        else:
            tol = 0.0015                            # media, mediana y rango son exactos
        nombres = {
            "media": "la media",
            "mediana": "la mediana",
            "rango": "el rango",
            "s": "la desviación estándar muestral (s)",
        }
        enunciado = (
            f"Se hicieron {n} mediciones de {nombre} (en {unidad}): "
            f"{', '.join(f'{t:g}' for t in datos)}. Calcula {nombres[pedir]}. Responde con 3 decimales."
        )
        pistas = {
            "s": [
                (v["sigma"], tol, "Parece que dividiste entre n. Como son mediciones de una muestra, la desviación muestral se divide entre n − 1."),
                (v["var"], tol, "Ese valor es la varianza (s²). Falta sacarle la raíz cuadrada."),
                (v["media"], tol, "Ese valor es la media, no la desviación estándar."),
            ],
            "media": [(v["mediana"], tol, "Ese valor es la mediana. La media se obtiene sumando todos los datos y dividiendo entre n.")],
            "mediana": [(v["media"], tol, "Ese valor es la media. La mediana es el dato central de los datos ya ordenados de menor a mayor.")],
            "rango": [(v["media"], tol, "Ese valor es la media. El rango es el dato mayor menos el dato menor.")],
        }[pedir]
        return {"enunciado": enunciado, "valor": valor, "tolerancia": tol, "pasos": res["pasos"], "pistas": pistas}

    if tema == "Probabilidad clásica":
        posibles = random.choice([10, 12, 20, 25, 40, 50])
        favorables = random.randint(1, posibles - 1)
        res = resolver_clasica(favorables, posibles)
        p = res["valores"]["p"]
        enunciado = (
            f"En una caja hay {posibles} piezas y {favorables} están defectuosas. Si se elige una al azar, "
            "¿cuál es la probabilidad de que sea defectuosa? Responde como decimal con 3 decimales (por ejemplo 0.250)."
        )
        pistas = [
            (p * 100, 0.05, "Diste el porcentaje. La pregunta pide la probabilidad como decimal entre 0 y 1."),
            (1 - p, 0.0015, "Ese valor es la probabilidad del complemento (que NO sea defectuosa)."),
            (favorables / max(posibles - favorables, 1), 0.0015, "Dividiste entre las piezas buenas. Se divide entre los casos posibles (todas las piezas)."),
        ]
        return {"enunciado": enunciado, "valor": p, "tolerancia": 0.0015, "pasos": res["pasos"], "pistas": pistas}

    if tema == "Distribución normal":
        mu = random.choice([50, 100, 150, 200])
        sigma = random.choice([4, 5, 8, 10])
        k = random.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
        a = round(mu + k * sigma, 2)
        tipo = random.choice(["<", ">"])
        res = resolver_normal(mu, sigma, tipo, a)
        p = res["valores"]["p"]
        enunciado = (
            f"La resistencia de un lote de resistores sigue una distribución normal con media {mu} Ω y "
            f"desviación estándar {sigma} Ω. Calcula P(X {tipo} {a:g}). Responde con 3 decimales."
        )
        pistas = [
            (1 - p, 0.002, "Revisa el sentido de la desigualdad: P(X < a) = Φ(z), pero P(X > a) = 1 − Φ(z)."),
            (res["valores"]["z"], 0.002, "Ese valor es z. Falta convertirlo en probabilidad con Φ(z) (la tabla de la normal estándar)."),
        ]
        return {"enunciado": enunciado, "valor": p, "tolerancia": 0.002, "pasos": res["pasos"], "pistas": pistas}

    # Regresión lineal: ley de Ohm
    R = random.choice([2.0, 3.0, 4.7, 5.0, 6.8, 10.0])
    corrientes = [0.2, 0.4, 0.6, 0.8, 1.0]
    voltajes = [round(R * i + random.uniform(-0.08, 0.08), 2) for i in corrientes]
    res = resolver_regresion(corrientes, voltajes, "A", "V")
    v = res["valores"]
    tol = max(0.0015, 0.002 * abs(v["m"]))
    enunciado = (
        "En un experimento se midió el voltaje V (en V) para distintas corrientes I (en A). "
        f"I: {', '.join(f'{t:g}' for t in corrientes)}; V: {', '.join(f'{t:g}' for t in voltajes)}. "
        "Ajusta la recta V = m·I + b y da la pendiente m (la resistencia, en Ω). Responde con 3 decimales."
    )
    pistas = [
        (1 / v["m"], max(0.0015, 0.005 / abs(v["m"])), "Invertiste x e y. La pendiente de V (eje y) contra I (eje x) es la resistencia: m = Sxy / Sxx con x = I."),
        (v["r"], 0.0015, "Ese valor es el coeficiente de correlación r, no la pendiente m."),
        (v["b"], 0.002, "Ese valor es el intercepto b. La pendiente es m = Sxy / Sxx."),
    ]
    return {"enunciado": enunciado, "valor": v["m"], "tolerancia": tol, "pasos": res["pasos"], "pistas": pistas}


def evaluar_respuesta(texto, problema):
    """Devuelve (estado, mensaje). estado: 'ok', 'mal' o 'invalida'."""
    t = texto.strip().replace(",", ".")
    if t == "":
        return "invalida", "Escribe tu respuesta antes de revisar."
    try:
        r = float(t)
        if not math.isfinite(r):
            raise ValueError
    except ValueError:
        return "invalida", "Escribe solo un número (por ejemplo 0.250)."
    if abs(r - problema["valor"]) <= problema["tolerancia"]:
        return "ok", f"¡Correcto! Tu respuesta {r:g} coincide con el resultado ({problema['valor']:.4f})."
    for valor, tol, mensaje in problema["pistas"]:
        if abs(r - valor) <= tol:
            return "mal", f"Todavía no. {mensaje}"
    return "mal", f"Todavía no: tu respuesta fue {r:g}. Revisa el procedimiento paso a paso y vuelve a intentarlo."


def seccion_practica():
    st.write(
        "Elige un tema y pulsa **Generar otro ejercicio**. Resuélvelo, escribe tu respuesta y pulsa "
        "**Revisar**: la app te dirá si está bien y, si no, te dará una pista sobre el error más común."
    )
    ss = st.session_state
    if "prac_ok" not in ss:
        ss["prac_ok"] = 0
        ss["prac_intentos"] = 0
        ss["prac_id"] = 0

    tema = st.radio("Tema", TEMAS_PRACTICA, horizontal=True, key="prac_tema")
    nuevo = st.button("Generar otro ejercicio", key="prac_nuevo")
    if nuevo or "prac" not in ss or ss["prac"]["tema"] != tema:
        problema = generar_problema(tema)
        problema["tema"] = tema
        problema["resultado"] = None
        problema["contado"] = False
        ss["prac"] = problema
        ss["prac_id"] += 1
    p = ss["prac"]

    st.subheader("Ejercicio")
    st.write(p["enunciado"])
    respuesta = st.text_input("Tu respuesta", key=f"prac_resp_{ss['prac_id']}")

    if st.button("Revisar", type="primary", key="prac_revisar"):
        estado, mensaje = evaluar_respuesta(respuesta, p)
        p["resultado"] = (estado, mensaje)
        if estado != "invalida" and not p["contado"]:
            p["contado"] = True
            ss["prac_intentos"] += 1
            if estado == "ok":
                ss["prac_ok"] += 1

    if p["resultado"] is not None:
        estado, mensaje = p["resultado"]
        if estado == "ok":
            st.success(mensaje)
        elif estado == "mal":
            st.error(mensaje)
        else:
            st.warning(mensaje)
        with st.expander("Ver la solución paso a paso"):
            mostrar_pasos(p["pasos"])
            st.write(f"**Respuesta correcta: {p['valor']:.4f}**")

    st.caption(f"Llevas {ss['prac_ok']} correctas de {ss['prac_intentos']} ejercicios revisados.")


def modulo_ejercicios():
    st.header("Ejercicios y práctica")
    parte = st.radio(
        "¿Qué quieres hacer?",
        ["Ejercicios resueltos (10)", "Practicar con retroalimentación"],
        horizontal=True,
        key="ej_parte",
    )
    if parte == "Ejercicios resueltos (10)":
        seccion_ejercicios_resueltos()
    else:
        seccion_practica()


# ---------------------------------------------------------------
# PÁGINA PRINCIPAL
# ---------------------------------------------------------------
st.title(NOMBRE_APP)
st.caption(f"Integrantes: {INTEGRANTES}")
st.caption(f"Asignatura: {ASIGNATURA} | Sección: {SECCION}")

st.sidebar.title("Módulos")
opcion = st.sidebar.radio(
    "Elige un módulo",
    [
        "Estadística descriptiva",
        "Tablas y gráficas",
        "Probabilidad y normal",
        "Correlación y regresión",
        "Ejercicios y práctica",
    ],
)

if opcion == "Estadística descriptiva":
    modulo_descriptiva()
elif opcion == "Tablas y gráficas":
    modulo_tablas_graficas()
elif opcion == "Probabilidad y normal":
    modulo_probabilidad()
elif opcion == "Correlación y regresión":
    modulo_regresion()
elif opcion == "Ejercicios y práctica":
    modulo_ejercicios()
