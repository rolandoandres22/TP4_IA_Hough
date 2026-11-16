import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def preprocesar_imagen(ruta_imagen: str, recortar_zoom_derecho: bool = True):
    """
    Carga la imagen, opcionalmente recorta la ampliación del aro para evitar
    que las flechas X/Y/Z del diagrama generen rectas falsas, y extrae bordes.
    """
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

    img_color = cv2.imread(ruta_imagen)

    # Si usás la Figura 2 del enunciado, recortamos el recuadro ampliado de la derecha
    # donde se ve claramente el anillo metálico (aro C) sin las flechas X, Y, Z.
    if recortar_zoom_derecho:
        h, w, _ = img_color.shape
        img_color = img_color[int(h * 0.35):int(h * 0.90), int(w * 0.62):int(w * 0.98)]

    gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (5, 5), 1.5)
    bordes = cv2.Canny(suavizada, threshold1=70, threshold2=160)

    return img_color, gris, bordes


# ==============================================================================
# ACTIVIDAD 2: TRANSFORMADA DE HOUGH PARA RECTAS (IMPLEMENTACIÓN EXPLÍCITA)
# ==============================================================================
def hough_rectas_manual(bordes: np.ndarray, umbral_votos: int = 65):
    alto, ancho = bordes.shape
    diag_max = int(np.ceil(np.sqrt(ancho**2 + alto**2)))

    thetas = np.deg2rad(np.arange(-90.0, 90.0, 1.0))
    rhos = np.linspace(-diag_max, diag_max, diag_max * 2)

    acumulador = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(bordes)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    for x, y in zip(x_idxs, y_idxs):
        for t_idx in range(len(thetas)):
            rho_idx = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag_max
            acumulador[rho_idx, t_idx] += 1

    rectas = []
    rho_idxs, theta_idxs = np.where(acumulador >= umbral_votos)
    for r_idx, t_idx in zip(rho_idxs, theta_idxs):
        rectas.append((rhos[r_idx], thetas[t_idx]))

    return rectas, acumulador


def dibujar_rectas(img_color: np.ndarray, rectas: list) -> np.ndarray:
    resultado = img_color.copy()
    for rho, theta in rectas:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        cv2.line(resultado, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return resultado


# ==============================================================================
# ACTIVIDAD 3: TRANSFORMADA DE HOUGH PARA CIRCUNFERENCIAS (RADIO CALIBRADO)
# ==============================================================================
def hough_circunferencia_optima(bordes: np.ndarray, radio_min: int = 20, radio_max: int = 55):
    """
    Evalúa el espacio acumulador (x0, y0) para el aro 'C' buscando el pico
    de máxima acumulación dentro de la tolerancia de escala del anillo.
    """
    alto, ancho = bordes.shape
    y_idxs, x_idxs = np.nonzero(bordes)

    angulos = np.deg2rad(np.arange(0, 360, 2))
    cos_a, sin_a = np.cos(angulos), np.sin(angulos)

    mejor_circulo = None
    max_votos_global = 0
    mejor_acumulador = None

    for r in range(radio_min, radio_max + 1):
        acumulador = np.zeros((alto, ancho), dtype=np.uint64)

        for x, y in zip(x_idxs, y_idxs):
            x0_cand = np.round(x - r * cos_a).astype(int)
            y0_cand = np.round(y - r * sin_a).astype(int)

            validos = (
                (x0_cand >= 0) & (x0_cand < ancho) &
                (y0_cand >= 0) & (y0_cand < alto)
            )
            for x0, y0 in zip(x0_cand[validos], y0_cand[validos]):
                acumulador[y0, x0] += 1

        y_max, x_max = np.unravel_index(np.argmax(acumulador), acumulador.shape)
        votos = int(acumulador[y_max, x_max])

        # Normalizamos ligeramente por el perímetro para no favorecer solo radios grandes
        score = votos / np.sqrt(r)
        if score > max_votos_global:
            max_votos_global = score
            mejor_circulo = (int(x_max), int(y_max), int(r), votos)
            mejor_acumulador = acumulador

    return [mejor_circulo] if mejor_circulo else [], mejor_acumulador


def dibujar_circunferencias(img_color: np.ndarray, circulos: list) -> np.ndarray:
    resultado = img_color.copy()
    for (x0, y0, r, votos) in circulos:
        # Aro "C" detectado (azul)
        cv2.circle(resultado, (x0, y0), r, (255, 0, 0), 2)
        # Centro "A" de montaje (punto rojo)
        cv2.circle(resultado, (x0, y0), 4, (0, 0, 255), -1)
        cv2.putText(
            resultado, f"A({x0},{y0})", (x0 + 8, y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2
        )
    return resultado


if __name__ == "__main__":
    CARPETA_RAIZ = os.path.dirname(os.path.abspath(__file__))
    RUTA_MOTOR = os.path.join(CARPETA_RAIZ, "motor_block.png")

    # Poné recortar_zoom_derecho=True si usás la Figura 2 del TP4,
    # o False si usás una foto frontal limpia de otro motor.
    img_orig, _, bordes = preprocesar_imagen(RUTA_MOTOR, recortar_zoom_derecho=True)

    # 1. Hough Rectas
    rectas, _ = hough_rectas_manual(bordes, umbral_votos=65)
    img_rectas = dibujar_rectas(img_orig, rectas)
    print(f"-> Rectas detectadas: {len(rectas)}")

    # 2. Hough Circunferencias (Aro C -> Centro A)
    circulos, _ = hough_circunferencia_optima(bordes, radio_min=20, radio_max=50)
    img_circulos = dibujar_circunferencias(img_orig, circulos)
    print(f"-> Centro A detectado (x0, y0, R, votos): {circulos}")

    # Guardar salidas para el informe
    cv2.imwrite(os.path.join(CARPETA_RAIZ, "salida_1_bordes_canny.jpg"), bordes)
    cv2.imwrite(os.path.join(CARPETA_RAIZ, "salida_2_hough_rectas.jpg"), img_rectas)
    cv2.imwrite(os.path.join(CARPETA_RAIZ, "salida_3_hough_circunferencia.jpg"), img_circulos)

    # Mostrar en pantalla
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    axs[0].imshow(bordes, cmap="gray")
    axs[0].set_title("1. Bordes Extraídos (Canny)")

    axs[1].imshow(cv2.cvtColor(img_rectas, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"2. Hough Rectas ({len(rectas)} detectadas)")

    axs[2].imshow(cv2.cvtColor(img_circulos, cv2.COLOR_BGR2RGB))
    axs[2].set_title("3. Hough Circunferencia (Centro A)")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()