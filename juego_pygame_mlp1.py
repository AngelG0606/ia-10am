import os
import csv
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.ion()

BASE_W, BASE_H = 1080, 720
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1


@dataclass
class Sample:
    velocidad_bala: float
    distancia: float

    salto: int

    altura_bala: int


class Juego:
    def __init__(self) -> None:
        pygame.init()

        self._flags = 0
        self._fullscreen = False

        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + agacharse + MLP")

        self.BLANCO = (255, 255, 255)
        self.NEGRO = (0, 0, 0)
        self.GRIS = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)

        self.corriendo = True
        self.modo_auto = False

        self.datos_modelo: List[Sample] = []
        self.modelo: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.modelo_entrenado = False
        self.clase_unica: Optional[int] = None
        self.ultima_proba_salto: Optional[float] = None

        # -----------------------------------------------------------------------
        # NUEVO: variable para rastrear la última predicción del modo auto.
        # Nos dice si el modelo decidió: 0=quieto, 1=saltar, 2=agacharse.
        # -----------------------------------------------------------------------
        self.ultima_accion_auto: Optional[int] = None

        self.decision_window = 500
        self.decision_record_every = 3
        self._decision_frame_counter = 0

        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size = (32, 48)
        self.bullet_size = (16, 16)
        self.ship_size = (64, 64)
        self.fondo_speed = 3

        self.salto = False
        self.en_suelo = True
        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0
        self.salto_vel = self.salto_vel_inicial

        # -----------------------------------------------------------------------
        # NUEVO: estado de agacharse.
        # agachado = True mientras el jugador mantiene presionada la tecla ABAJO.
        # -----------------------------------------------------------------------
        self.agachado = False

        # -----------------------------------------------------------------------
        # NUEVO: altura del jugador cuando está de pie y cuando está agachado.
        # Al agacharse la hitbox baja a la mitad para poder esquivar balas altas.
        # -----------------------------------------------------------------------
        self.player_height_normal = 48   # altura en píxeles (base, se escala)
        self.player_height_agachado = 24  # mitad de la altura normal

        self.current_frame = 0
        self.frame_speed = 10
        self.frame_count = 0

        self.velocidad_bala = -12
        self.bala_disparada = False
        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        self.bala_altura_tipo: int = 0

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    # ----------------- resolución / assets -----------------
    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)

        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin = int(50 * self.scale)
        ground_offset = int(100 * self.scale)
        self.ground_y = self.h - ground_offset

        self.player_size = (int(32 * self.scale), int(48 * self.scale))
        self.bullet_size = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size = (int(64 * self.scale), int(64 * self.scale))
        self.fondo_speed = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 15 * self.scale
        self.gravedad = 1 * self.scale
        self.salto_vel = self.salto_vel_inicial

        # -----------------------------------------------------------------------
        # NUEVO: recalculamos las alturas del jugador según la escala actual.
        # -----------------------------------------------------------------------
        self.player_height_normal = int(48 * self.scale)
        self.player_height_agachado = int(24 * self.scale)  # mitad de la normal

        self.decision_window = int(500 * self.scale)

        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(
                self.margin, self.ground_y,
                self.player_size[0], self.player_size[1]
            )
            self.bala = pygame.Rect(
                self.w - self.margin,
                self.ground_y - self.player_height_normal // 2,
                self.bullet_size[0],
                self.bullet_size[1],
            )
            self.nave = pygame.Rect(
                self.w - int(100 * self.scale),
                self.ground_y,
                self.ship_size[0],
                self.ship_size[1],
            )

    def _cargar_assets(self) -> None:
        def safe_load(path: str, size: Tuple[int, int], fallback_color=(200, 200, 200, 255)) -> pygame.Surface:
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill(fallback_color)
                return surf

        base = os.path.dirname(__file__)
        self.jugador_frames = [
            safe_load(os.path.join(base, "assets/sprites/mono_frame_1.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_2.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_3.png"), self.player_size),
            safe_load(os.path.join(base, "assets/sprites/mono_frame_4.png"), self.player_size),
        ]

        agachado_size = (self.player_size[0], self.player_height_agachado)
        self.jugador_agachado_img = safe_load(
            os.path.join(base, "assets/sprites/mono_agachado.png"),
            agachado_size,
            (180, 180, 255, 255),   # color fallback azulado para distinguirlo
        )

        self.bala_img = safe_load(
            os.path.join(base, "assets/sprites/purple_ball.png"),
            self.bullet_size,
            (160, 120, 255, 255),
        )
        self.fondo_img = safe_load(
            os.path.join(base, "assets/game/fondo2.png"),
            (self.w, self.h),
            (40, 40, 40, 255),
        )
        self.nave_img = safe_load(
            os.path.join(base, "assets/game/ufo.png"),
            self.ship_size,
            (140, 255, 200, 255),
        )

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            w = info.current_w or self.w
            h = info.current_h or self.h
            self.pantalla = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            self._apply_resolution(w, h, reset_positions=True)
        else:
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    # ----------------- estado juego / modelo -----------------
    def _reset_estado_juego(self) -> None:
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
        self.bala.x = self.w - self.margin
        self.bala.y = self.ground_y - self.player_height_normal // 2
        self.bala_disparada = False
        self.velocidad_bala = int(-10 * self.scale)
        self.bala_altura_tipo = 0
        self.salto = False
        self.en_suelo = True
        self.salto_vel = self.salto_vel_inicial
        self.agachado = False
        self._restaurar_hitbox_normal()   # asegura tamaño correcto del Rect
        self._decision_frame_counter = 0
        self.fondo_x1 = 0
        self.fondo_x2 = self.w

    def _reset_modelo(self) -> None:
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica = None


    def _restaurar_hitbox_normal(self) -> None:
        """Ajusta el Rect del jugador a su tamaño de pie."""
        self.jugador.height = self.player_height_normal
        # El jugador siempre "pisa" el suelo desde su borde inferior.
        self.jugador.y = self.ground_y

    def _aplicar_hitbox_agachado(self) -> None:
        """
        Reduce la hitbox del jugador a la mitad de su altura.
        Para que el personaje quede 'pegado al suelo' (no flote),
        ajustamos Y de modo que el borde inferior siga en ground_y.
        """
        self.jugador.height = self.player_height_agachado
        # Borde inferior = ground_y + player_height_normal  →  y = ground_y + diferencia
        diferencia = self.player_height_normal - self.player_height_agachado
        self.jugador.y = self.ground_y + diferencia

    # ----------------- export / gráficas -----------------
    def exportar_datos_csv(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para exportar."
        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # ---------------------------------------------------------------
                # NUEVO: el CSV ahora incluye la columna altura_bala (0=baja, 1=alta).
                # ---------------------------------------------------------------
                writer.writerow(["velocidad_bala", "distancia", "altura_bala", "salto"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.altura_bala, s.salto])
        except Exception as e:
            return f"Error al guardar CSV: {e}"
        return f"CSV guardado en datos_mlp.csv ({len(self.datos_modelo)} filas)."

    def graficar_datos_2d(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para graficar."
        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        colores = {0: "blue", 1: "red", 2: "green"}
        cs = [colores.get(s.salto, "gray") for s in self.datos_modelo]

        fig_num = plt.figure("Datos MLP - 2D", figsize=(8, 6)).number
        plt.figure(fig_num)
        plt.clf()
        ax = plt.gca()
        # -----------------------------------------------------------------------
        #  el tamaño del punto indica la altura de la bala:
        #   punto grande (s=60) = bala alta  (tipo 1 → agacharse)
        #   punto pequeño (s=20) = bala baja (tipo 0 → saltar)
        # Esto permite ver visualmente cómo se distribuyen ambos tipos.
        # -----------------------------------------------------------------------
        sizes = [80 if s.altura_bala == 1 else 20 for s in self.datos_modelo]
        ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=sizes)
        ax.set_xlabel("Distancia jugador-bala")
        ax.set_ylabel("Velocidad bala")
        ax.set_title("Datos MLP\nazul=quieto  rojo=salto  verde=agachado\npunto grande=bala alta  pequeño=bala baja")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        return "Mostrando gráfica 2D interactiva."

    def graficar_datos_3d(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para graficar."
        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        zs = list(range(len(self.datos_modelo)))
        colores = {0: "blue", 1: "red", 2: "green"}
        cs = [colores.get(s.salto, "gray") for s in self.datos_modelo]

        fig = plt.figure("Datos MLP - 3D", figsize=(8, 6))
        plt.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia")
        ax.set_ylabel("Velocidad bala")
        ax.set_zlabel("Índice (tiempo aproximado)")
        ax.set_title("Datos MLP 3D (azul=quieto, rojo=salto, verde=agachado)")
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()
        return "Mostrando gráfica 3D interactiva."

    # ----------------- bala / salto -----------------
    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            self.bala_altura_tipo = random.randint(0, 1)

            if self.bala_altura_tipo == 0:
                # Bala BAJA: viaja a la altura de los pies/piernas del jugador.
                # El jugador de pie la toca (borde inferior de su hitbox).
                # Al SALTAR sube y la bala pasa por debajo.
                # ground_y es la esquina superior del jugador, así que sus pies
                # están en ground_y + player_height_normal. Ponemos la bala
                # justo en la parte baja del personaje.
                self.bala.y = self.ground_y + int(self.player_height_normal * 0.75)
            else:
                # Bala ALTA: viaja a la altura del torso/cabeza del jugador.
                # El jugador de pie la toca (parte superior de su hitbox).
                # Al AGACHARSE su hitbox se reduce a la mitad inferior,
                # así la bala pasa por encima sin tocarlo.
                self.bala.y = self.ground_y + int(self.player_height_normal * 0.05)

            self.bala_disparada = True

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False

    def iniciar_salto(self) -> None:
        if self.en_suelo and not self.agachado:
            self.salto = True
            self.en_suelo = False

    def manejar_salto(self) -> None:
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            if self.jugador.y >= self.ground_y:
                self.jugador.y = self.ground_y
                self.salto = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True

    # -----------------------------------------------------------------------
    # NUEVO: métodos para iniciar y terminar la acción de agacharse.
    # -----------------------------------------------------------------------
    def iniciar_agacharse(self) -> None:
        """
        Activa el estado agachado si el jugador está en el suelo y no está saltando.
        Cambia la hitbox a la versión pequeña para que las balas altas pasen por encima.
        """
        if self.en_suelo and not self.salto:
            self.agachado = True
            self._aplicar_hitbox_agachado()

    def terminar_agacharse(self) -> None:
        """
        Desactiva el estado agachado y restaura la hitbox normal del jugador.
        Se llama cuando el jugador suelta la tecla de agacharse.
        """
        if self.agachado:
            self.agachado = False
            self._restaurar_hitbox_normal()

    # ----------------- datos / ML -----------------
    def registrar_decision_manual(self) -> None:
        if not self.bala_disparada:
            return
        distancia = abs(self.jugador.x - self.bala.x)
        # -----------------------------------------------------------------------
        # La etiqueta tiene 3 clases:
        #   0 = jugador en el suelo y de pie (quieto)
        #   1 = jugador en el aire (saltando)
        #   2 = jugador agachado
        # -----------------------------------------------------------------------
        if self.agachado:
            salto_label = 2
        elif not self.en_suelo:
            salto_label = 1
        else:
            salto_label = 0

        self.datos_modelo.append(
            Sample(
                velocidad_bala=float(self.velocidad_bala),
                distancia=float(distancia),
                salto=salto_label,
                # ---------------------------------------------------------------
                # NUEVO: guardamos también el tipo de altura de este disparo
                # (0=baja, 1=alta) como feature extra para el modelo.
                # Sin este dato el MLP no tiene forma de distinguir si una bala
                # que viene a cierta distancia requiere saltar o agacharse.
                # ---------------------------------------------------------------
                altura_bala=self.bala_altura_tipo,
            )
        )

    def entrenar_modelo(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas más datos (>= 80). Juega en MANUAL."
        # -----------------------------------------------------------------------
        # NUEVO: ahora el vector de entrada tiene 3 features en lugar de 2:
        #   [velocidad_bala, distancia, altura_bala]
        # El tercer feature (altura_bala: 0=baja, 1=alta) es clave para que
        # el modelo sepa qué acción tomar ante balas de distinto nivel.
        # -----------------------------------------------------------------------
        X = [[s.velocidad_bala, s.distancia, s.altura_bala] for s in samples]
        y = [s.salto for s in samples]
        clases = sorted(set(y))
        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            nombres = {0: "QUIETO", 1: "SIEMPRE SALTA", 2: "SIEMPRE AGACHADO"}
            tipo = nombres.get(self.clase_unica, str(self.clase_unica))
            return True, f"Modelo trivial: {tipo}. Junta datos de varias clases."
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = MLPClassifier(
            hidden_layer_sizes=(8, 8),   # un poco más grande para 3 clases
            activation="relu",
            solver="adam",
            max_iter=300000,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        self._reset_modelo()
        self.scaler = scaler
        self.modelo = clf
        self.modelo_entrenado = True
        return True, f"MLP (3 clases) entrenado. Accuracy test ≈ {acc:.3f}"

    def decision_auto(self) -> int:
        """
        Consulta el modelo y devuelve la acción recomendada:
          0 = no hacer nada
          1 = saltar
          2 = agacharse
        Si no hay modelo, devuelve 0 (quieto).
        """
        if not self.modelo_entrenado or not self.bala_disparada:
            return 0

        # Caso especial: modelo trivial de una sola clase
        if self.clase_unica is not None and self.modelo is None:
            return self.clase_unica

        if self.modelo is None or self.scaler is None:
            return 0

        distancia = abs(self.jugador.x - self.bala.x)
        X = [[float(self.velocidad_bala), float(distancia), float(self.bala_altura_tipo)]]
        Xs = self.scaler.transform(X)
        pred = int(self.modelo.predict(Xs)[0])

        # Guardamos la probabilidad de salto para mostrarla en pantalla
        if hasattr(self.modelo, "predict_proba"):
            probas = self.modelo.predict_proba(Xs)[0]
            # Las clases pueden estar en orden variable; buscamos la clase 1 (salto)
            clases_lista = list(self.modelo.classes_)
            idx_salto = clases_lista.index(1) if 1 in clases_lista else None
            self.ultima_proba_salto = float(probas[idx_salto]) if idx_salto is not None else None

        self.ultima_accion_auto = pred
        return pred
    
    def decision_auto_saltar(self) -> bool:
        return self.decision_auto() == 1

    # ----------------- menú -----------------
    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(60 * self.scale)))

        opciones = [
            "M - Manual (reinicia dataset y borra modelo)",
            "A - Auto (usa MLP; sin modelo NO salta)",
            "T - Entrenar MLP",
            "C - Exportar datos a CSV",
            "F - Fullscreen (toggle)",
            "Q - Salir",
            "",
            "Controles en juego:",
            "  ESPACIO  → saltar",
            "  ABAJO (↓) → agacharse (mantener pulsado)",
        ]

        x0 = int(80 * self.scale)
        y = int(140 * self.scale)
        line_h = self.fuente.get_linesize()
        pad = max(6, int(6 * self.scale))
        for op in opciones:
            t = self.fuente.render(op, True, self.BLANCO)
            self.pantalla.blit(t, (x0, y))
            y += line_h + pad

        y += int(8 * self.scale)
        estado = [
            f"Memoria: {len(self.datos_modelo)} | Modelo: {'sí' if self.modelo_entrenado else 'no'}",
            f"Resolución: {self.w}x{self.h} | scale≈{self.scale:.2f}",
        ]
        for line in estado:
            t = self.fuente_chica.render(line, True, self.GRIS)
            self.pantalla.blit(t, (x0, y))
            y += self.fuente_chica.get_linesize()

        if msg:
            mm = self.fuente_chica.render(msg, True, self.AMARILLO)
            self.pantalla.blit(mm, (x0, y + int(12 * self.scale)))

        pygame.display.flip()

    def mostrar_menu(self) -> None:
        msg = ""
        esperando = True
        self._decision_frame_counter = 0
        while esperando and self.corriendo:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    esperando = False
                    break
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.datos_modelo.clear()
                        self._reset_modelo()
                        self._reset_estado_juego()
                        esperando = False
                        break
                    if e.key == pygame.K_a:
                        if not self.modelo_entrenado:
                            msg = "Primero entrena el MLP (T) en esta sesión."
                        else:
                            self.modo_auto = True
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_t:
                        ok, info = self.entrenar_modelo()
                        msg = info if ok else f"Error: {info}"
                    if e.key == pygame.K_c:
                        msg = self.exportar_datos_csv()
                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

    # ----------------- render / loop -----------------
    def _update_frame(self) -> None:
        self.fondo_x1 -= self.fondo_speed
        self.fondo_x2 -= self.fondo_speed
        if self.fondo_x1 <= -self.w:
            self.fondo_x1 = self.w
        if self.fondo_x2 <= -self.w:
            self.fondo_x2 = self.w
        self.pantalla.blit(self.fondo_img, (self.fondo_x1, 0))
        self.pantalla.blit(self.fondo_img, (self.fondo_x2, 0))

        self.frame_count += 1
        if self.frame_count >= self.frame_speed:
            self.current_frame = (self.current_frame + 1) % len(self.jugador_frames)
            self.frame_count = 0
        if self.agachado:
            # Dibujamos en la posición Y actual del Rect (ya ajustada por _aplicar_hitbox_agachado)
            self.pantalla.blit(self.jugador_agachado_img, (self.jugador.x, self.jugador.y))
        else:
            self.pantalla.blit(
                self.jugador_frames[self.current_frame],
                (self.jugador.x, self.jugador.y)
            )

        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        if self.bala_disparada:
            self.bala.x += self.velocidad_bala
        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()
        self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))

        if self.jugador.colliderect(self.bala):
            self.agachado = False
            self._reset_estado_juego()

        # Info del modelo en tiempo real
        if self.modelo_entrenado and self.modo_auto:
            lineas_info = []
            if self.ultima_proba_salto is not None:
                lineas_info.append(f"proba_salto≈{self.ultima_proba_salto:.2f}")
            nombres_accion = {0: "quieto", 1: "salta", 2: "agacha"}
            if self.ultima_accion_auto is not None:
                lineas_info.append(f"accion={nombres_accion.get(self.ultima_accion_auto, '?')}")
            for i, linea in enumerate(lineas_info):
                txt = self.fuente_chica.render(linea, True, self.AMARILLO)
                self.pantalla.blit(txt, (10, 10 + i * self.fuente_chica.get_linesize()))
        if self.bala_disparada:
            if self.bala_altura_tipo == 1:
                hint_txt = self.fuente_chica.render("↑ AGÁCHATE", True, (255, 160, 60))
            else:
                hint_txt = self.fuente_chica.render("↓ SALTA", True, (60, 220, 255))
            # Centramos el texto en la parte superior de la pantalla
            hint_x = self.w // 2 - hint_txt.get_width() // 2
            self.pantalla.blit(hint_txt, (hint_x, int(12 * self.scale)))

    def loop(self) -> None:
        reloj = pygame.time.Clock()
        self.mostrar_menu()

        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        self.corriendo = False
                    elif e.key in (pygame.K_ESCAPE, pygame.K_p):
                        self.agachado = False          # limpiamos estado al salir
                        self._reset_estado_juego()
                        self.mostrar_menu()
                    elif e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    elif e.key == pygame.K_SPACE and not self.modo_auto and self.en_suelo:
                        self.iniciar_salto()

                    elif e.key == pygame.K_DOWN and not self.modo_auto:
                        self.iniciar_agacharse()

                elif e.type == pygame.KEYUP:
                    if e.key == pygame.K_DOWN and not self.modo_auto:
                        self.terminar_agacharse()

            if not self.corriendo:
                break

            if self.modo_auto:
                accion = self.decision_auto()
                if accion == 1:
                    # El modelo dice: saltar
                    self.terminar_agacharse()   # por si estaba agachado
                    self.iniciar_salto()
                elif accion == 2:
                    # El modelo dice: agacharse
                    self.iniciar_agacharse()
                else:
                    # El modelo dice: no hacer nada → de pie
                    self.terminar_agacharse()
            else:
                self.registrar_decision_manual()

            if self.salto:
                self.manejar_salto()

            if not self.bala_disparada:
                self.disparar_bala()

            self._update_frame()
            pygame.display.flip()
            reloj.tick(45)

        pygame.quit()


def main() -> None:
    Juego().loop()


if __name__ == "__main__":
    main()
