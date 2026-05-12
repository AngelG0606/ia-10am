import os
import csv
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


BASE_W, BASE_H = 1080, 720
EXTRA_SCALE = 1.1
FPS = 45


@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    tiempo_impacto: float
    altura_bala: float
    estado_anterior: float
    frames_estado: float
    accion: int


class Juego:
    def __init__(self) -> None:
        pygame.init()

        self._flags = 0
        self._fullscreen = False
        self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + MLP corregido")

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

        self.ultima_proba_salto = 0.0
        self.ultima_proba_agachar = 0.0

        self.w, self.h = BASE_W, BASE_H
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100

        self.player_size = (32, 48)
        self.player_size_agachado = (32, 24)
        self.bullet_size = (16, 16)
        self.ship_size = (64, 64)

        self.fondo_speed = 3

        self.salto = False
        self.agachado = False
        self.en_suelo = True
        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0
        self.salto_vel = self.salto_vel_inicial

        self.crouch_timer = 0
        self.crouch_duration = 12

        self.current_frame = 0
        self.frame_speed = 10
        self.frame_count = 0

        self.velocidad_bala = -12
        self.bala_disparada = False
        self.altura_bala_relativa = 0.0

        self.fondo_x1 = 0
        self.fondo_x2 = BASE_W

        self.estado_modelo = 0
        self.frames_estado_modelo = 0

        self._accion_manual_frame = 0
        self._tecla_abajo_presionada = False

        self.frames_compensacion = 8
        self.distancia_decision_max = 620
        self.distancia_decision_min = 20

        self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)
        self.scale = max(1.0, min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE)

        self.margin = int(50 * self.scale)
        self.ground_y = self.h - int(100 * self.scale)

        self.player_size = (int(32 * self.scale), int(48 * self.scale))
        self.player_size_agachado = (int(32 * self.scale), int(24 * self.scale))
        self.bullet_size = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size = (int(64 * self.scale), int(64 * self.scale))

        self.fondo_speed = max(1, int(2 * self.scale))
        self.salto_vel_inicial = 15 * self.scale
        self.gravedad = 1 * self.scale
        self.salto_vel = self.salto_vel_inicial

        self.bala_altura_media = int(self.player_size[1] // 2)

        self.distancia_decision_max = int(620 * self.scale)
        self.distancia_decision_min = int(20 * self.scale)

        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(
                self.margin,
                self.ground_y,
                self.player_size[0],
                self.player_size[1],
            )

            self.bala = pygame.Rect(
                self.w - self.margin,
                self.ground_y + int(10 * self.scale),
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
            os.path.join(base, "assets/game/real-UFO.png"),
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

    def _reset_estado_juego(self) -> None:
        self.jugador.x = self.margin
        self.jugador.y = self.ground_y
        self.jugador.width = self.player_size[0]
        self.jugador.height = self.player_size[1]

        self.nave.x = self.w - int(100 * self.scale)
        self.nave.y = self.ground_y

        self.bala.x = self.w - self.margin
        self.bala.y = self.ground_y + int(10 * self.scale)

        self.bala_disparada = False
        self.altura_bala_relativa = 0.0
        self.velocidad_bala = int(-10 * self.scale)

        self.salto = False
        self.agachado = False
        self.en_suelo = True
        self.salto_vel = self.salto_vel_inicial
        self.crouch_timer = 0

        self.estado_modelo = 0
        self.frames_estado_modelo = 0

        self._accion_manual_frame = 0
        self._tecla_abajo_presionada = False

        self.fondo_x1 = 0
        self.fondo_x2 = self.w

    def _reset_modelo(self) -> None:
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica = None
        self.ultima_proba_salto = 0.0
        self.ultima_proba_agachar = 0.0

    def _estado_fisico_actual(self) -> int:
        if not self.en_suelo and self.salto:
            return 1
        if self.agachado:
            return 2
        return 0

    def _tiempo_impacto(self, distancia: float) -> float:
        return distancia / max(1.0, abs(float(self.velocidad_bala)))

    def _features_desde_estado(
        self,
        velocidad_bala: float,
        distancia: float,
        altura_bala: float,
        estado_anterior: float,
        frames_estado: float,
    ) -> List[float]:
        tiempo_impacto = distancia / max(1.0, abs(float(velocidad_bala)))

        h0 = 1.0 if altura_bala == 0.0 else 0.0
        h1 = 1.0 if altura_bala == 1.0 else 0.0

        e0 = 1.0 if estado_anterior == 0.0 else 0.0
        e1 = 1.0 if estado_anterior == 1.0 else 0.0
        e2 = 1.0 if estado_anterior == 2.0 else 0.0

        return [
            float(velocidad_bala),
            float(distancia),
            float(tiempo_impacto),
            h0,
            h1,
            e0,
            e1,
            e2,
            min(float(frames_estado), 60.0),
        ]

    def exportar_datos_csv(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para exportar."

        ruta = os.path.join(os.path.dirname(__file__), "datos_mlp.csv")

        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "velocidad_bala",
                    "distancia",
                    "tiempo_impacto",
                    "altura_bala",
                    "estado_anterior",
                    "frames_estado",
                    "accion",
                ])

                for s in self.datos_modelo:
                    writer.writerow([
                        s.velocidad_bala,
                        s.distancia,
                        s.tiempo_impacto,
                        s.altura_bala,
                        s.estado_anterior,
                        s.frames_estado,
                        s.accion,
                    ])

        except Exception as e:
            return f"Error al guardar CSV: {e}"

        return f"CSV guardado en datos_mlp.csv ({len(self.datos_modelo)} filas)."

    def cargar_datos_csv(self) -> str:
        ruta = os.path.join(os.path.dirname(__file__), "datos_mlp.csv")

        if not os.path.exists(ruta):
            return "El archivo datos_mlp.csv no existe."

        try:
            nuevos_datos = []

            with open(ruta, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    if "accion" in row:
                        accion = int(row["accion"])
                    elif "salto" in row:
                        accion = int(row["salto"])
                    else:
                        continue

                    velocidad = float(row["velocidad_bala"])
                    distancia = float(row["distancia"])
                    tiempo = float(row.get("tiempo_impacto") or distancia / max(1.0, abs(velocidad)))
                    altura = float(row.get("altura_bala", 0.0))
                    estado = float(row.get("estado_anterior", 0.0))
                    frames = float(row.get("frames_estado", 0.0))

                    nuevos_datos.append(
                        Sample(
                            velocidad_bala=velocidad,
                            distancia=distancia,
                            tiempo_impacto=tiempo,
                            altura_bala=altura,
                            estado_anterior=estado,
                            frames_estado=frames,
                            accion=accion,
                        )
                    )

            self.datos_modelo = nuevos_datos
            self._reset_modelo()

            return f"Cargadas {len(self.datos_modelo)} filas de datos_mlp.csv. Entrena otra vez con T."

        except Exception as e:
            return f"Error al cargar CSV: {e}"

    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            self.altura_bala_relativa = random.choice([0.0, 1.0])

            if self.altura_bala_relativa == 0.0:
                self.bala.y = self.ground_y + int(10 * self.scale)
            else:
                self.bala.y = self.ground_y - self.bala_altura_media + int(14 * self.scale)

            self.bala_disparada = True
            self.estado_modelo = self._estado_fisico_actual()
            self.frames_estado_modelo = 0

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False

    def iniciar_salto(self) -> bool:
        if self.en_suelo and not self.agachado:
            self.salto = True
            self.en_suelo = False
            return True
        return False

    def iniciar_agacharse(self) -> bool:
        if self.en_suelo and not self.salto:
            if not self.agachado:
                self.agachado = True
                self.crouch_timer = 0
                self.jugador.height = self.player_size_agachado[1]
                self.jugador.y = self.ground_y + (self.player_size[1] - self.player_size_agachado[1])
            return True
        return False

    def iniciar_agacharse_auto_sostenido(self) -> bool:
        if self.en_suelo and not self.salto:
            if not self.agachado:
                self.agachado = True
                self.jugador.height = self.player_size_agachado[1]
                self.jugador.y = self.ground_y + (self.player_size[1] - self.player_size_agachado[1])

            self.crouch_timer = self.crouch_duration
            return True

        return False

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
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad

            if self.jugador.y >= self.ground_y:
                self.jugador.y = self.ground_y
                self.jugador.height = self.player_size[1]
                self.salto = False
                self.salto_vel = self.salto_vel_inicial
                self.en_suelo = True

    def _reescribir_frames_previos(self, accion: int) -> None:
        for i in range(1, min(self.frames_compensacion + 1, len(self.datos_modelo) + 1)):
            idx = len(self.datos_modelo) - i
            s = self.datos_modelo[idx]

            if s.accion != 0:
                break

            if s.altura_bala != self.altura_bala_relativa:
                break

            if s.distancia > self.distancia_decision_max:
                continue

            s.accion = accion

    def registrar_decision_manual(self) -> None:
        if not self.bala_disparada or self.bala.x < self.jugador.x:
            self._accion_manual_frame = 0
            return

        distancia = abs(self.jugador.x - self.bala.x)
        tiempo_impacto = self._tiempo_impacto(distancia)

        estado_ant = self._estado_fisico_actual()
        frames_ant = getattr(self, "frames_estado_modelo", 0)

        accion = 0

        if self.distancia_decision_min <= distancia <= self.distancia_decision_max:
            if self._accion_manual_frame == 1:
                accion = 1
                self._reescribir_frames_previos(1)

            elif self._accion_manual_frame == 2 or self._tecla_abajo_presionada:
                accion = 2

                if self._accion_manual_frame == 2:
                    self._reescribir_frames_previos(2)

        self.datos_modelo.append(
            Sample(
                velocidad_bala=float(self.velocidad_bala),
                distancia=float(distancia),
                tiempo_impacto=float(tiempo_impacto),
                altura_bala=float(self.altura_bala_relativa),
                estado_anterior=float(estado_ant),
                frames_estado=float(frames_ant),
                accion=accion,
            )
        )

        estado_actual = self._estado_fisico_actual()

        if estado_actual == getattr(self, "estado_modelo", 0):
            self.frames_estado_modelo = min(frames_ant + 1, 60)
        else:
            self.estado_modelo = estado_actual
            self.frames_estado_modelo = 1

        self._accion_manual_frame = 0

    def entrenar_modelo(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)

        if len(samples) < 120:
            return False, "Necesitas mas datos (>= 120). Juega en MANUAL esquivando varias balas."

        X_orig = [
            self._features_desde_estado(
                s.velocidad_bala,
                s.distancia,
                s.altura_bala,
                s.estado_anterior,
                s.frames_estado,
            )
            for s in samples
        ]

        y_orig = [s.accion for s in samples]
        clases = sorted(set(y_orig))

        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            return True, "Modelo trivial: solo hay una clase. Necesitas ejemplos de saltar/agachar."

        conteo = {c: y_orig.count(c) for c in clases}

        if any(conteo[c] < 5 for c in clases):
            return False, f"Muy pocos ejemplos por clase: {conteo}. Juega mas antes de entrenar."

        max_muestras = max(conteo.values())

        X_bal = []
        y_bal = []

        for c in clases:
            indices_c = [i for i, val in enumerate(y_orig) if val == c]

            for idx in random.choices(indices_c, k=max_muestras):
                X_bal.append(X_orig[idx])
                y_bal.append(y_orig[idx])

        X_train, X_test, y_train, y_test = train_test_split(
            X_bal,
            y_bal,
            test_size=0.2,
            random_state=42,
            stratify=y_bal,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            max_iter=3000,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42,
        )

        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        self._reset_modelo()
        self.scaler = scaler
        self.modelo = clf
        self.modelo_entrenado = True

        return True, f"MLP entrenado. Accuracy test aprox {acc:.3f} | clases={conteo}"

    def decision_auto(self) -> int:
        if not self.modelo_entrenado or not self.bala_disparada:
            return 0

        distancia = abs(self.jugador.x - self.bala.x)

        if distancia > self.distancia_decision_max or distancia < self.distancia_decision_min:
            return 0

        if self.clase_unica is not None and self.modelo is None:
            return int(self.clase_unica)

        if self.modelo is None or self.scaler is None:
            return 0

        estado_actual = self._estado_fisico_actual()
        frames_estado = getattr(self, "frames_estado_modelo", 0)

        X = [
            self._features_desde_estado(
                self.velocidad_bala,
                distancia,
                self.altura_bala_relativa,
                estado_actual,
                frames_estado,
            )
        ]

        Xs = self.scaler.transform(X)
        accion = int(self.modelo.predict(Xs)[0])

        self.ultima_proba_salto = 0.0
        self.ultima_proba_agachar = 0.0

        if hasattr(self.modelo, "predict_proba"):
            probas = self.modelo.predict_proba(Xs)[0]
            clases = list(self.modelo.classes_)

            if 1 in clases:
                self.ultima_proba_salto = float(probas[clases.index(1)])

            if 2 in clases:
                self.ultima_proba_agachar = float(probas[clases.index(2)])

            if self.altura_bala_relativa == 0.0 and self.ultima_proba_salto >= 0.38:
                accion = 1

            elif self.altura_bala_relativa == 1.0 and self.ultima_proba_agachar >= 0.38:
                accion = 2

        return accion

    def _actualizar_estado_modelo_auto(self) -> None:
        estado_actual = self._estado_fisico_actual()

        if estado_actual == getattr(self, "estado_modelo", 0):
            self.frames_estado_modelo = min(getattr(self, "frames_estado_modelo", 0) + 1, 60)
        else:
            self.estado_modelo = estado_actual
            self.frames_estado_modelo = 1

    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)

        titulo = self.fuente.render("MENU", True, self.BLANCO)
        self.pantalla.blit(
            titulo,
            (self.w // 2 - titulo.get_width() // 2, int(60 * self.scale)),
        )

        opciones = [
            "M - Manual (reinicia dataset y borra modelo)",
            "A - Auto (usa MLP; sin modelo NO juega)",
            "T - Entrenar MLP",
            "C - Exportar datos a CSV",
            "L - Cargar datos desde CSV",
            "F - Fullscreen",
            "Q - Salir",
        ]

        x0 = int(80 * self.scale)
        y = int(140 * self.scale)

        for op in opciones:
            t = self.fuente.render(op, True, self.BLANCO)
            self.pantalla.blit(t, (x0, y))
            y += self.fuente.get_linesize() + max(6, int(6 * self.scale))

        y += int(8 * self.scale)

        estado = [
            f"Memoria: {len(self.datos_modelo)} | Modelo: {'si' if self.modelo_entrenado else 'no'}",
            f"Resolucion: {self.w}x{self.h} | scale aprox {self.scale:.2f}",
            "El MLP aprende cuando INICIAR la accion, no solo cuando ya esta saltando.",
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
                            msg = "Primero entrena el MLP con T."
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

                    if e.key == pygame.K_l:
                        msg = self.cargar_datos_csv()

                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()

                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

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

        jugador_sprite = self.jugador_frames[self.current_frame]

        if self.agachado:
            sprite_agachado = pygame.transform.scale(jugador_sprite, self.player_size_agachado)
            self.pantalla.blit(sprite_agachado, (self.jugador.x, self.jugador.y))
        else:
            self.pantalla.blit(jugador_sprite, (self.jugador.x, self.jugador.y))

        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        if self.bala_disparada:
            self.bala.x += self.velocidad_bala

        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()

            if self.modo_auto and self.agachado:
                self.terminar_agacharse()

        self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))

        if self.modelo_entrenado and self.modo_auto:
            distancia = abs(self.jugador.x - self.bala.x)
            t_imp = self._tiempo_impacto(distancia)

            info = (
                f"p salto={self.ultima_proba_salto:.2f} | "
                f"p agacha={self.ultima_proba_agachar:.2f} | "
                f"t={t_imp:.1f}"
            )

            txt = self.fuente_chica.render(info, True, self.AMARILLO)
            self.pantalla.blit(txt, (int(20 * self.scale), int(20 * self.scale)))

        if self.jugador.colliderect(self.bala):
            self._reset_estado_juego()

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
                        if self.iniciar_salto():
                            self._accion_manual_frame = 1

                    elif e.key == pygame.K_DOWN and not self.modo_auto:
                        if self.iniciar_agacharse():
                            self._accion_manual_frame = 2
                            self._tecla_abajo_presionada = True

                elif e.type == pygame.KEYUP:
                    if e.key == pygame.K_DOWN and not self.modo_auto:
                        self._tecla_abajo_presionada = False
                        self.terminar_agacharse()

            if not self.corriendo:
                break

            if self.modo_auto:
                accion = self.decision_auto()

                if accion == 1 and self.en_suelo and not self.agachado:
                    self.iniciar_salto()

                elif accion == 2 and self.en_suelo and not self.salto:
                    self.iniciar_agacharse_auto_sostenido()

                elif accion != 2 and self.agachado:
                    self.terminar_agacharse()

                self._actualizar_estado_modelo_auto()

            else:
                self.registrar_decision_manual()

            if self.salto:
                self.manejar_salto()

            self.manejar_agacharse()

            if not self.bala_disparada:
                self.disparar_bala()

            self._update_frame()
            pygame.display.flip()
            reloj.tick(FPS)

        pygame.quit()


def main() -> None:
    Juego().loop()


if __name__ == "__main__":
    main()
