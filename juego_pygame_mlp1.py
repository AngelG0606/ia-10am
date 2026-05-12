import os
import csv
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Opcional: para graficar los datos en 2D y 3D
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

# ── FIX 1: Cuántos frames de margen adelantarse para agacharse (bala alta)
CROUCH_LEAD_FRAMES = 30   # el personaje se agachará cuando queden ~30 frames para el impacto


@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    altura_bala: float
    tiempo_impacto: float   # FIX 2: nueva feature — frames estimados al impacto
    accion: int             # 0=nada, 1=saltar, 2=agacharse


class Juego:
    def __init__(self) -> None:
        pygame.init()

        self._flags = 0
        self._fullscreen = False

        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + MLP (solo memoria)")

        self.BLANCO = (255, 255, 255)
        self.NEGRO  = (0, 0, 0)
        self.GRIS   = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)
        self.VERDE  = (100, 255, 140)

        self.corriendo  = True
        self.modo_auto  = False

        self.datos_modelo: List[Sample] = []
        self.modelo:  Optional[MLPClassifier]   = None
        self.scaler:  Optional[StandardScaler]  = None
        self.modelo_entrenado = False
        self.clase_unica: Optional[int] = None
        self.ultima_proba_salto: Optional[float] = None

        # ── FIX 3: registrar cada N frames, no todos (reduce ruido y dominio de clase 0)
        self.decision_record_every = 4
        self._decision_frame_counter = 0

        self.w, self.h = start_w, start_h
        self.scale  = 1.0
        self.margin = 50
        self.ground_y    = self.h - 100
        self.player_size = (32, 48)
        self.bullet_size = (16, 16)
        self.ship_size   = (64, 64)
        self.fondo_speed = 3

        self.salto     = False
        self.agachado  = False
        self.en_suelo  = True
        self.salto_vel_inicial = 15.0
        self.gravedad  = 1.0
        self.salto_vel = self.salto_vel_inicial

        self.crouch_timer    = 0
        # ── FIX 4: duración del agachado automático aumentada
        self.crouch_duration = 20

        self.current_frame = 0
        self.frame_speed   = 10
        self.frame_count   = 0

        self.velocidad_bala     = -12
        self.bala_disparada     = False
        self.altura_bala_relativa = 0.0
        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    # ──────────────────────────────────────────────────────────────
    # Resolución / assets
    # ──────────────────────────────────────────────────────────────
    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)
        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin       = int(50 * self.scale)
        ground_offset     = int(100 * self.scale)
        self.ground_y     = self.h - ground_offset

        self.player_size          = (int(32 * self.scale), int(48 * self.scale))
        self.player_size_agachado = (int(32 * self.scale), int(24 * self.scale))
        self.bullet_size  = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size    = (int(64 * self.scale), int(64 * self.scale))
        self.fondo_speed  = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 15 * self.scale
        self.gravedad          = 1  * self.scale
        self.salto_vel         = self.salto_vel_inicial
        self.bala_altura_media = int(self.player_size[1] // 2)

        self.decision_window = int(500 * self.scale)
        self.crouch_duration = 20   # FIX 4: siempre 20 frames independiente de escala

        self.fuente       = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(
                self.margin, self.ground_y,
                self.player_size[0], self.player_size[1]
            )
            self.bala = pygame.Rect(
                self.w - self.margin,
                self.ground_y + int(10 * self.scale),
                self.bullet_size[0], self.bullet_size[1],
            )
            self.nave = pygame.Rect(
                self.w - int(100 * self.scale),
                self.ground_y,
                self.ship_size[0], self.ship_size[1],
            )

    def _cargar_assets(self) -> None:
        def safe_load(path, size, fallback_color=(200, 200, 200, 255)):
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
        self.bala_img = safe_load(
            os.path.join(base, "assets/sprites/purple_ball.png"),
            self.bullet_size, (160, 120, 255, 255),
        )
        self.fondo_img = safe_load(
            os.path.join(base, "assets/game/fondo2.png"),
            (self.w, self.h), (40, 40, 40, 255),
        )
        self.nave_img = safe_load(
            os.path.join(base, "assets/game/ufo.png"),
            self.ship_size, (140, 255, 200, 255),
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

    # ──────────────────────────────────────────────────────────────
    # Estado juego / modelo
    # ──────────────────────────────────────────────────────────────
    def _reset_estado_juego(self) -> None:
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.jugador.height = self.player_size[1]
        self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
        self.bala.x = self.w - self.margin
        self.bala.y = self.ground_y + int(10 * self.scale)
        self.bala_disparada     = False
        self.altura_bala_relativa = 0.0
        self.velocidad_bala     = int(-10 * self.scale)
        self.salto    = False
        self.agachado = False
        self.en_suelo = True
        self.salto_vel = self.salto_vel_inicial
        self.crouch_timer = 0
        self._decision_frame_counter = 0
        self.fondo_x1 = 0
        self.fondo_x2 = self.w

    def _reset_modelo(self) -> None:
        self.modelo  = None
        self.scaler  = None
        self.modelo_entrenado = False
        self.clase_unica = None

    # ──────────────────────────────────────────────────────────────
    # Export / gráficas
    # ──────────────────────────────────────────────────────────────
    def exportar_datos_csv(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para exportar."
        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["velocidad_bala", "distancia", "altura_bala",
                                  "tiempo_impacto", "accion"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.altura_bala,
                                     s.tiempo_impacto, s.accion])
        except Exception as e:
            return f"Error al guardar CSV: {e}"
        return f"CSV guardado en datos_mlp.csv ({len(self.datos_modelo)} filas)."

    def graficar_datos_2d(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para graficar."
        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        color_map = {0: "blue", 1: "red", 2: "green"}
        cs = [color_map.get(s.accion, "gray") for s in self.datos_modelo]

        fig_num = plt.figure("Datos MLP - 2D", figsize=(8, 6)).number
        plt.figure(fig_num); plt.clf()
        ax = plt.gca()
        ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia jugador-bala")
        ax.set_ylabel("Velocidad bala")
        ax.set_title("Datos MLP (azul=nada, rojo=salto, verde=agacha)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show(block=False); plt.draw()
        return "Mostrando gráfica 2D interactiva."

    def graficar_datos_3d(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para graficar."
        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        zs = [s.altura_bala for s in self.datos_modelo]
        color_map = {0: "blue", 1: "red", 2: "green"}
        cs = [color_map.get(s.accion, "gray") for s in self.datos_modelo]

        fig = plt.figure("Datos MLP - 3D", figsize=(8, 6)); plt.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia")
        ax.set_ylabel("Velocidad bala")
        ax.set_zlabel("Altura bala")
        ax.set_title("Datos MLP 3D (azul=nada, rojo=salto, verde=agacha)")
        plt.tight_layout(); plt.show(block=False); plt.draw()
        return "Mostrando gráfica 3D interactiva."

    # ──────────────────────────────────────────────────────────────
    # Física: bala / salto / agacharse
    # ──────────────────────────────────────────────────────────────
    def _calcular_tiempo_impacto(self) -> float:
        """FIX 2: frames estimados hasta que la bala alcance al jugador."""
        if not self.bala_disparada or self.velocidad_bala == 0:
            return 999.0
        distancia = abs(self.jugador.x - self.bala.x)
        return distancia / abs(self.velocidad_bala)

    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            self.altura_bala_relativa = random.choice([0.0, 1.0])
            if self.altura_bala_relativa == 0.0:
                self.bala.y = self.ground_y + int(10 * self.scale)
            else:
                self.bala.y = self.ground_y - self.bala_altura_media + int(14 * self.scale)
            self.bala_disparada = True

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False

    def iniciar_salto(self) -> None:
        if self.en_suelo and not self.agachado:
            self.salto    = True
            self.en_suelo = False

    def iniciar_agacharse(self) -> None:
        """Agachado MANUAL: sin timer, dura hasta KEYUP."""
        if self.en_suelo and not self.salto and not self.agachado:
            self.agachado = True
            self.crouch_timer = 0
            self.jugador.height = self.player_size_agachado[1]
            self.jugador.y = self.ground_y + (self.player_size[1] - self.player_size_agachado[1])

    def iniciar_agacharse_auto(self) -> None:
        """Agachado AUTO corto (bala baja): ciclos agacha→levanta."""
        if self.en_suelo and not self.salto and not self.agachado:
            self.agachado = True
            self.crouch_timer = self.crouch_duration
            self.jugador.height = self.player_size_agachado[1]
            self.jugador.y = self.ground_y + (self.player_size[1] - self.player_size_agachado[1])

    def iniciar_agacharse_auto_sostenido(self) -> None:
        """Agachado AUTO sostenido (bala alta): reinicia el timer cada frame."""
        if self.en_suelo and not self.salto:
            if not self.agachado:
                self.agachado = True
                self.jugador.height = self.player_size_agachado[1]
                self.jugador.y = self.ground_y + (self.player_size[1] - self.player_size_agachado[1])
            self.crouch_timer = self.crouch_duration   # mantener agachado

    def terminar_agacharse(self) -> None:
        if self.agachado:
            self.agachado = False
            self.crouch_timer = 0
            self.jugador.height = self.player_size[1]
            self.jugador.y = self.ground_y

    def manejar_agacharse(self) -> None:
        if self.agachado and self.crouch_timer > 0:
            self.crouch_timer -= 1
            if self.crouch_timer <= 0:
                self.terminar_agacharse()

    def manejar_salto(self) -> None:
        if self.salto:
            self.jugador.y  -= int(self.salto_vel)
            self.salto_vel  -= self.gravedad
            if self.jugador.y >= self.ground_y:
                self.jugador.y   = self.ground_y
                self.jugador.height = self.player_size[1]
                self.salto    = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True

    # ──────────────────────────────────────────────────────────────
    # Datos / ML
    # ──────────────────────────────────────────────────────────────
    def registrar_decision_manual(self) -> None:
        """
        FIX 3 + FIX 5:
        - Solo registra cada `decision_record_every` frames (reduce ruido).
        - Solo registra cuando la bala está dentro de la ventana de decisión.
        - Incluye la nueva feature tiempo_impacto.
        """
        if not self.bala_disparada:
            return

        self._decision_frame_counter += 1
        if self._decision_frame_counter % self.decision_record_every != 0:
            return

        distancia = abs(self.jugador.x - self.bala.x)

        # FIX 5: descartar muestras de cuando la bala está muy lejos
        if distancia > self.decision_window:
            return

        tiempo_impacto = self._calcular_tiempo_impacto()

        if not self.en_suelo and self.salto:
            accion = 1
        elif self.agachado:
            accion = 2
        else:
            accion = 0

        self.datos_modelo.append(
            Sample(
                velocidad_bala  = float(self.velocidad_bala),
                distancia       = float(distancia),
                altura_bala     = float(self.altura_bala_relativa),
                tiempo_impacto  = float(tiempo_impacto),
                accion          = accion,
            )
        )

    def entrenar_modelo(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas más datos (>= 80). Juega en MANUAL."

        X = [
            [s.velocidad_bala, s.distancia, s.altura_bala, s.tiempo_impacto]
            for s in samples
        ]
        y = [s.accion for s in samples]

        clases = sorted(set(y))
        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            nombres = {0: "NADA", 1: "SIEMPRE SALTA", 2: "SIEMPRE AGACHA"}
            return True, (
                f"Modelo trivial: {nombres.get(self.clase_unica, '?')}. "
                "Varía tus acciones para mejor modelo."
            )

        # ── FIX 1: Balancear clases con oversampling (clase mayoritaria marca el tamaño)
        conteo   = Counter(y)
        max_count = max(conteo.values())
        X_bal, y_bal = [], []
        for clase in clases:
            idx = [i for i, yi in enumerate(y) if yi == clase]
            Xi  = [X[i] for i in idx]
            yi  = [y[i] for i in idx]
            if len(Xi) < max_count:
                Xi, yi = resample(Xi, yi, n_samples=max_count, random_state=42)
            X_bal.extend(Xi)
            y_bal.extend(yi)

        X, y = X_bal, y_bal

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        self._reset_modelo()
        self.scaler = scaler
        self.modelo = clf
        self.modelo_entrenado = True

        conteo_str = " | ".join(f"clase {k}:{v}" for k, v in sorted(Counter(y_bal).items()))
        return True, f"MLP entrenado. Accuracy ≈ {acc:.3f} | {conteo_str}"

    def decision_auto(self) -> int:
        """Retorna 0=nada, 1=saltar, 2=agacharse."""
        if not self.modelo_entrenado or not self.bala_disparada:
            return 0

        distancia      = abs(self.jugador.x - self.bala.x)
        tiempo_impacto = self._calcular_tiempo_impacto()

        if self.clase_unica is not None and self.modelo is None:
            self.ultima_proba_salto = 1.0 if self.clase_unica == 1 else 0.0
            return int(self.clase_unica)

        if self.modelo is None or self.scaler is None:
            return 0

        X  = [[float(self.velocidad_bala), float(distancia),
               float(self.altura_bala_relativa), float(tiempo_impacto)]]
        Xs = self.scaler.transform(X)
        accion = int(self.modelo.predict(Xs)[0])

        if hasattr(self.modelo, "predict_proba"):
            probas = self.modelo.predict_proba(Xs)[0]
            clases = list(self.modelo.classes_)
            self.ultima_proba_salto = float(probas[clases.index(1)]) if 1 in clases else 0.0
        else:
            self.ultima_proba_salto = 1.0 if accion == 1 else 0.0

        return accion

    # ──────────────────────────────────────────────────────────────
    # Menú
    # ──────────────────────────────────────────────────────────────
    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ", True, self.BLANCO)
        self.pantalla.blit(titulo, (self.w // 2 - titulo.get_width() // 2, int(60 * self.scale)))

        opciones = [
            "M - Manual (reinicia dataset y borra modelo)",
            "A - Auto (usa MLP; sin modelo NO salta)",
            "T - Entrenar MLP",
            "G - Gráfica 2D",
            "H - Gráfica 3D",
            "C - Exportar datos a CSV",
            "F - Fullscreen (toggle)",
            "Q - Salir",
        ]
        x0 = int(80 * self.scale)
        y  = int(140 * self.scale)
        line_h = self.fuente.get_linesize()
        pad    = max(6, int(6 * self.scale))
        for op in opciones:
            t = self.fuente.render(op, True, self.BLANCO)
            self.pantalla.blit(t, (x0, y))
            y += line_h + pad

        y += int(8 * self.scale)
        conteo_txt = ""
        if self.datos_modelo:
            conteo = Counter(s.accion for s in self.datos_modelo)
            conteo_txt = f" [nada:{conteo[0]} salto:{conteo[1]} agacha:{conteo[2]}]"
        estado = [
            f"Memoria: {len(self.datos_modelo)}{conteo_txt}",
            f"Modelo: {'SÍ' if self.modelo_entrenado else 'no'} | "
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
                    if e.key == pygame.K_g:
                        msg = self.graficar_datos_2d()
                    if e.key == pygame.K_h:
                        msg = self.graficar_datos_3d()
                    if e.key == pygame.K_c:
                        msg = self.exportar_datos_csv()
                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

    # ──────────────────────────────────────────────────────────────
    # Render / loop
    # ──────────────────────────────────────────────────────────────
    def _update_frame(self) -> None:
        # Fondo
        self.fondo_x1 -= self.fondo_speed
        self.fondo_x2 -= self.fondo_speed
        if self.fondo_x1 <= -self.w:
            self.fondo_x1 = self.w
        if self.fondo_x2 <= -self.w:
            self.fondo_x2 = self.w
        self.pantalla.blit(self.fondo_img, (self.fondo_x1, 0))
        self.pantalla.blit(self.fondo_img, (self.fondo_x2, 0))

        # Animación del jugador
        self.frame_count += 1
        if self.frame_count >= self.frame_speed:
            self.current_frame = (self.current_frame + 1) % len(self.jugador_frames)
            self.frame_count = 0

        jugador_sprite = self.jugador_frames[self.current_frame]
        if self.agachado:
            sprite_agachado = pygame.transform.scale(jugador_sprite, self.player_size_agachado)
            self.pantalla.blit(sprite_agachado, (self.jugador.x, self.jugador.y))
        else:
            self.pantalla.blit(jugador_sprite, (self.jugador.x, self.jugador.y))

        # Nave y bala
        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        if self.bala_disparada:
            self.bala.x += self.velocidad_bala
        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()
            if self.modo_auto and self.agachado:
                self.terminar_agacharse()
        self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))

        # Colisión → reset
        if self.jugador.colliderect(self.bala):
            self._reset_estado_juego()

        # HUD
        y_hud = 10
        if self.agachado:
            txt = self.fuente_chica.render("[AGACHADO]", True, (100, 200, 255))
            self.pantalla.blit(txt, (10, y_hud))
            y_hud += self.fuente_chica.get_linesize() + 2

        if self.bala_disparada:
            tipo_bala  = "Bala ALTA (agáchate)" if self.altura_bala_relativa == 1.0 else "Bala BAJA (salta)"
            color_tip  = (255, 120, 80) if self.altura_bala_relativa == 1.0 else (80, 220, 120)
            txt = self.fuente_chica.render(tipo_bala, True, color_tip)
            self.pantalla.blit(txt, (10, y_hud))
            y_hud += self.fuente_chica.get_linesize() + 2

            # FIX info: tiempo estimado al impacto en HUD
            ti = self._calcular_tiempo_impacto()
            txt_ti = self.fuente_chica.render(f"tiempo_impacto≈{ti:.1f}f", True, self.GRIS)
            self.pantalla.blit(txt_ti, (10, y_hud))
            y_hud += self.fuente_chica.get_linesize() + 2

        if self.modelo_entrenado and self.modo_auto and self.ultima_proba_salto is not None:
            txt = self.fuente_chica.render(
                f"proba_salto≈{self.ultima_proba_salto:.2f}", True, self.AMARILLO
            )
            self.pantalla.blit(txt, (10, y_hud))
            y_hud += self.fuente_chica.get_linesize() + 2

        # Contador de muestras (útil en modo manual para saber cuándo entrenar)
        if not self.modo_auto:
            conteo = Counter(s.accion for s in self.datos_modelo)
            txt_n = self.fuente_chica.render(
                f"muestras n:{conteo[0]} s:{conteo[1]} a:{conteo[2]}", True, self.VERDE
            )
            self.pantalla.blit(txt_n, (10, y_hud))

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
                        self._reset_estado_juego()
                        self.mostrar_menu()
                    elif e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    elif e.key == pygame.K_SPACE and not self.modo_auto:
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
                if accion == 1 and self.en_suelo and not self.agachado:
                    self.iniciar_salto()
                elif accion == 2 and self.en_suelo and not self.salto:
                    # FIX 4: siempre usar sostenido para ambos tipos de bala —
                    # el modelo ya aprendió cuándo agacharse con tiempo_impacto
                    self.iniciar_agacharse_auto_sostenido()
            else:
                self.registrar_decision_manual()

            if self.salto:
                self.manejar_salto()

            self.manejar_agacharse()

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