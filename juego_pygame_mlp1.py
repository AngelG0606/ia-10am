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
    altura_bala: float
    frame_bala: int
    tipo_bala: int
    frames_agachado_actual: int
    frames_quieto_actual: int
    frames_desde_ultimo_agachado: int


class Juego:
    def __init__(self) -> None:
        pygame.init()

        self._flags = 0
        self._fullscreen = False

        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + agacharse + quieto + MLP")

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
        self.ultima_proba_agacharse: Optional[float] = None
        self.ultima_accion_auto: Optional[int] = None
        self.ultima_accion_corregida_auto: Optional[int] = None

        self.decision_record_every = 1
        self._decision_frame_counter = 0

        self._quieto_frame_counter = 0
        self.QUIETO_SKIP_RATIO = 6

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

        self.agachado = False
        self.tecla_abajo_presionada = False

        self.auto_agachado_timer = 0
        self.auto_agachado_cooldown = 0
        self.AUTO_AGACHADO_DURACION = 8
        self.AUTO_AGACHADO_DESCANSO = 5

        self.frames_agachado_actual = 0
        self.frames_quieto_actual = 0
        self.frames_desde_ultimo_agachado = 999

        self.player_height_normal = 48
        self.player_height_agachado = 24

        self.current_frame = 0
        self.frame_speed = 10
        self.frame_count = 0

        self.velocidad_bala = -12
        self.bala_disparada = False
        self.frames_bala_actual = 0
        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        self.bala_altura_tipo: int = 0
        self.bala_altura_feature: float = 0.0

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

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

        self.player_height_normal = int(48 * self.scale)
        self.player_height_agachado = int(24 * self.scale)

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
            (180, 180, 255, 255),
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

    def _reset_estado_juego(self) -> None:
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
        self.bala.x = self.w - self.margin
        self.bala.y = self.ground_y - self.player_height_normal // 2

        self.bala_disparada = False
        self.frames_bala_actual = 0
        self.velocidad_bala = int(-10 * self.scale)
        self.bala_altura_tipo = 0
        self.bala_altura_feature = float(self.bala.y)

        self.salto = False
        self.en_suelo = True
        self.salto_vel = self.salto_vel_inicial

        self.agachado = False
        self.tecla_abajo_presionada = False
        self.auto_agachado_timer = 0
        self.auto_agachado_cooldown = 0
        self.frames_agachado_actual = 0
        self.frames_quieto_actual = 0
        self.frames_desde_ultimo_agachado = 999
        self._restaurar_hitbox_normal()

        self._decision_frame_counter = 0
        self._quieto_frame_counter = 0

        self.fondo_x1 = 0
        self.fondo_x2 = self.w

    def _reset_modelo(self) -> None:
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica = None
        self.ultima_proba_salto = None
        self.ultima_proba_agacharse = None
        self.ultima_accion_auto = None
        self.ultima_accion_corregida_auto = None

    def _restaurar_hitbox_normal(self) -> None:
        self.jugador.height = self.player_height_normal
        self.jugador.y = self.ground_y

    def _aplicar_hitbox_agachado(self) -> None:
        self.jugador.height = self.player_height_agachado
        diferencia = self.player_height_normal - self.player_height_agachado
        self.jugador.y = self.ground_y + diferencia

    def exportar_datos_csv(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para exportar."

        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")

        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "velocidad_bala",
                    "distancia",
                    "altura_bala",
                    "frame_bala",
                    "tipo_bala",
                    "frames_agachado_actual",
                    "frames_quieto_actual",
                    "frames_desde_ultimo_agachado",
                    "salto",
                ])
                for s in self.datos_modelo:
                    writer.writerow([
                        s.velocidad_bala,
                        s.distancia,
                        s.altura_bala,
                        s.frame_bala,
                        s.tipo_bala,
                        s.frames_agachado_actual,
                        s.frames_quieto_actual,
                        s.frames_desde_ultimo_agachado,
                        s.salto,
                    ])
        except Exception as e:
            return f"Error al guardar CSV: {e}"

        return f"CSV guardado en datos_mlp.csv ({len(self.datos_modelo)} filas)."

    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)

            self.bala_altura_tipo = random.choices(
                [0, 1, 2, 3],
                weights=[4, 4, 3, 4],
                k=1,
            )[0]

            self.frames_bala_actual = 0

            if self.bala_altura_tipo == 0:
                self.bala.y = self.ground_y + int(self.player_height_normal * 0.45)
            elif self.bala_altura_tipo == 1:
                self.bala.y = self.ground_y + int(self.player_height_normal * 0.05)
            elif self.bala_altura_tipo == 2:
                self.bala.y = self.ground_y - self.bullet_size[1] - int(14 * self.scale)
            else:
                self.bala.y = self.ground_y + int(self.player_height_normal * 0.18)

            self.bala_altura_feature = float(self.bala.y)
            self.bala_disparada = True

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False
        self.frames_bala_actual = 0

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

    def iniciar_agacharse(self) -> None:
        if self.en_suelo and not self.salto:
            self.agachado = True
            self._aplicar_hitbox_agachado()

    def terminar_agacharse(self) -> None:
        if self.agachado:
            self.agachado = False
            self._restaurar_hitbox_normal()

    def actualizar_contadores_accion(self) -> None:
        if self.agachado:
            self.frames_agachado_actual += 1
            self.frames_quieto_actual = 0
            self.frames_desde_ultimo_agachado = 0
        elif self.en_suelo and not self.salto:
            self.frames_quieto_actual += 1
            self.frames_agachado_actual = 0
            self.frames_desde_ultimo_agachado = min(999, self.frames_desde_ultimo_agachado + 1)
        else:
            self.frames_agachado_actual = 0
            self.frames_quieto_actual = 0
            self.frames_desde_ultimo_agachado = min(999, self.frames_desde_ultimo_agachado + 1)

    def actualizar_agachado_auto_multiple(self, accion: int) -> None:
        if accion not in (2, 3):
            self.auto_agachado_timer = 0
            self.auto_agachado_cooldown = 0
            self.terminar_agacharse()
            return

        if self.salto or not self.en_suelo:
            self.terminar_agacharse()
            return

        if self.auto_agachado_timer > 0:
            self.iniciar_agacharse()
            self.auto_agachado_timer -= 1

            if self.auto_agachado_timer <= 0:
                self.terminar_agacharse()
                self.auto_agachado_cooldown = self.AUTO_AGACHADO_DESCANSO

            return

        if self.auto_agachado_cooldown > 0:
            self.terminar_agacharse()
            self.auto_agachado_cooldown -= 1
            return

        self.iniciar_agacharse()
        self.auto_agachado_timer = self.AUTO_AGACHADO_DURACION

    def corregir_accion_por_tipo_bala(self, accion: int) -> int:
        """
        Corrige confusiones entre mantener y pulsar.
        El modelo decide si debe agacharse, pero el tipo de bala decide
        si esa agachada es mantenida o multiple.
        """
        if self.bala_altura_tipo == 0:
            return 1 if accion == 1 else 0

        if self.bala_altura_tipo == 1:
            return 2 if accion in (2, 3) else 0

        if self.bala_altura_tipo == 2:
            return 0

        if self.bala_altura_tipo == 3:
            return 3 if accion in (2, 3) else 0

        return accion

    def actualizar_agachado_auto(self, accion: int) -> None:
        if accion == 3:
            self.actualizar_agachado_auto_multiple(accion)
            return

        if accion == 2 and self.en_suelo and not self.salto:
            self.auto_agachado_timer = 0
            self.auto_agachado_cooldown = 0
            self.iniciar_agacharse()
            return

        self.auto_agachado_timer = 0
        self.auto_agachado_cooldown = 0
        self.terminar_agacharse()

    def actualizar_agachado_manual_continuo(self) -> None:
        teclas = pygame.key.get_pressed()
        abajo_presionado = teclas[pygame.K_DOWN]

        self.tecla_abajo_presionada = abajo_presionado

        if abajo_presionado and self.en_suelo and not self.salto:
            self.iniciar_agacharse()
        else:
            self.terminar_agacharse()

    def registrar_decision_manual(self) -> None:
        if not self.bala_disparada:
            return

        self._decision_frame_counter += 1

        if self._decision_frame_counter % self.decision_record_every != 0:
            return

        distancia = abs(self.jugador.x - self.bala.x)

        if self.tecla_abajo_presionada and self.en_suelo and not self.salto:
            if self.bala_altura_tipo == 3:
                salto_label = 3
            else:
                salto_label = 2
        elif not self.en_suelo:
            salto_label = 1
        else:
            salto_label = 0

        if salto_label == 0 and self.bala_altura_tipo != 2 and distancia > self.w * 0.40:
            self._quieto_frame_counter += 1
            if self._quieto_frame_counter % self.QUIETO_SKIP_RATIO != 0:
                return
        else:
            self._quieto_frame_counter = 0

        self.datos_modelo.append(
            Sample(
                velocidad_bala=float(self.velocidad_bala),
                distancia=float(distancia),
                salto=salto_label,
                altura_bala=float(self.bala_altura_feature),
                frame_bala=self.frames_bala_actual,
                tipo_bala=self.bala_altura_tipo,
                frames_agachado_actual=self.frames_agachado_actual,
                frames_quieto_actual=self.frames_quieto_actual,
                frames_desde_ultimo_agachado=self.frames_desde_ultimo_agachado,
            )
        )

    def _balancear_dataset(self, X: list, y: list) -> Tuple[list, list]:
        """
        Balancea por (tipo_bala, clase), no solo por clase.
        Esto evita que PULSA ABAJO domine sobre MANTEN ABAJO.
        """
        grupos = {}

        for xi, yi in zip(X, y):
            tipo_bala = int(xi[4])
            clave = (tipo_bala, yi)
            grupos.setdefault(clave, []).append((xi, yi))

        if len(grupos) < 2:
            return X, y

        max_grupo = max(len(items) for items in grupos.values())

        balanceados = []
        for items in grupos.values():
            balanceados.extend(items)
            faltan = max_grupo - len(items)
            if faltan > 0:
                balanceados.extend(random.choices(items, k=faltan))

        random.shuffle(balanceados)

        X_bal = [xi for xi, _ in balanceados]
        y_bal = [yi for _, yi in balanceados]

        return X_bal, y_bal

    def entrenar_modelo(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)

        if len(samples) < 80:
            return False, "Necesitas más datos (>= 80). Juega en MANUAL."

        X = [
            [
                s.velocidad_bala,
                s.distancia,
                s.altura_bala,
                s.frame_bala,
                s.tipo_bala,
                s.frames_agachado_actual,
                s.frames_quieto_actual,
                s.frames_desde_ultimo_agachado,
            ]
            for s in samples
        ]
        y = [s.salto for s in samples]

        clases = sorted(set(y))
        conteo_orig = Counter(y)

        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            nombres = {
                0: "QUIETO",
                1: "SIEMPRE SALTA",
                2: "SIEMPRE MANTIENE AGACHADO",
                3: "SIEMPRE PULSA AGACHADO",
            }
            tipo = nombres.get(self.clase_unica, str(self.clase_unica))
            return True, f"Modelo trivial: {tipo}. Junta datos de varias clases."

        for clase in clases:
            if conteo_orig[clase] < 2:
                return False, (
                    "Necesitas al menos 2 ejemplos por clase para entrenar. "
                    f"Conteo: quieto={conteo_orig.get(0, 0)}, "
                    f"salto={conteo_orig.get(1, 0)}, "
                    f"mantener={conteo_orig.get(2, 0)}, "
                    f"pulso={conteo_orig.get(3, 0)}."
                )

        X, y = self._balancear_dataset(X, y)
        conteo_bal = Counter(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = MLPClassifier(
            hidden_layer_sizes=(32, 32),
            activation="relu",
            solver="adam",
            max_iter=6000,
            random_state=42,
        )

        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)

        self._reset_modelo()
        self.scaler = scaler
        self.modelo = clf
        self.modelo_entrenado = True

        detalle = "  ".join(
            f"cl{c}:{conteo_orig[c]}->{conteo_bal.get(c, 0)}"
            for c in sorted(conteo_orig)
        )

        return True, f"MLP entrenado. Acc≈{acc:.3f}  [{detalle}]"

    def decision_auto(self) -> int:
        if not self.modelo_entrenado or not self.bala_disparada:
            return 0

        if self.clase_unica is not None and self.modelo is None:
            return self.clase_unica

        if self.modelo is None or self.scaler is None:
            return 0

        distancia = abs(self.jugador.x - self.bala.x)

        X = [[
            float(self.velocidad_bala),
            float(distancia),
            float(self.bala_altura_feature),
            float(self.frames_bala_actual),
            float(self.bala_altura_tipo),
            float(self.frames_agachado_actual),
            float(self.frames_quieto_actual),
            float(self.frames_desde_ultimo_agachado),
        ]]

        Xs = self.scaler.transform(X)
        pred = int(self.modelo.predict(Xs)[0])

        if hasattr(self.modelo, "predict_proba"):
            probas = self.modelo.predict_proba(Xs)[0]
            clases_lista = list(self.modelo.classes_)

            idx_salto = clases_lista.index(1) if 1 in clases_lista else None
            idx_agacharse = clases_lista.index(2) if 2 in clases_lista else None

            self.ultima_proba_salto = float(probas[idx_salto]) if idx_salto is not None else None
            self.ultima_proba_agacharse = float(probas[idx_agacharse]) if idx_agacharse is not None else None

        self.ultima_accion_auto = pred
        accion_corregida = self.corregir_accion_por_tipo_bala(pred)
        self.ultima_accion_corregida_auto = accion_corregida
        return accion_corregida

    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)

        titulo = self.fuente.render("MENÚ", True, self.BLANCO)
        self.pantalla.blit(
            titulo,
            (self.w // 2 - titulo.get_width() // 2, int(60 * self.scale)),
        )

        opciones = [
            "M - Manual (reinicia dataset y borra modelo)",
            "A - Auto (usa MLP; sin modelo NO salta)",
            "T - Entrenar MLP",
            "C - Exportar datos a CSV",
            "F - Fullscreen (toggle)",
            "Q - Salir",
            "",
            "Controles:",
            "ESPACIO - saltar",
            "ABAJO - mantener o pulsar segun el hint",
            "QUIETO - no presiones nada",
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

        conteo = Counter(s.salto for s in self.datos_modelo)

        estado = [
            f"Memoria: {len(self.datos_modelo)} | Modelo: {'sí' if self.modelo_entrenado else 'no'}",
            f"quieto:{conteo.get(0, 0)}  salto:{conteo.get(1, 0)}  mantener:{conteo.get(2, 0)}  pulso:{conteo.get(3, 0)}",
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
            self.pantalla.blit(self.jugador_agachado_img, (self.jugador.x, self.jugador.y))
        else:
            self.pantalla.blit(
                self.jugador_frames[self.current_frame],
                (self.jugador.x, self.jugador.y),
            )

        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        if self.bala_disparada:
            self.bala.x += self.velocidad_bala

        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()

        self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))

        if self.jugador.colliderect(self.bala):
            self.agachado = False
            self.tecla_abajo_presionada = False
            self._reset_estado_juego()

        if self.modelo_entrenado and self.modo_auto:
            nombres_accion = {
                0: "quieto",
                1: "salta",
                2: "mantiene",
                3: "pulsa",
            }

            if self.ultima_accion_auto is not None:
                cruda = nombres_accion.get(self.ultima_accion_auto, "?")
                corregida = nombres_accion.get(self.ultima_accion_corregida_auto, "?")
                linea = f"modelo={cruda} final={corregida}"
                txt = self.fuente_chica.render(linea, True, self.AMARILLO)
                self.pantalla.blit(txt, (10, 10))

        if self.bala_disparada:
            if self.bala_altura_tipo == 0:
                hint_txt = self.fuente_chica.render("↑ SALTA", True, (60, 220, 255))
            elif self.bala_altura_tipo == 1:
                hint_txt = self.fuente_chica.render("↓ MANTEN ABAJO", True, (255, 160, 60))
            elif self.bala_altura_tipo == 2:
                hint_txt = self.fuente_chica.render("QUIETO", True, (120, 255, 120))
            else:
                hint_txt = self.fuente_chica.render("↓↓ PULSA ABAJO", True, (255, 90, 210))

            hint_x = self.w // 2 - hint_txt.get_width() // 2
            self.pantalla.blit(hint_txt, (hint_x, int(12 * self.scale)))

        if not self.modo_auto:
            conteo = Counter(s.salto for s in self.datos_modelo)
            info_datos = (
                f"Datos - quieto:{conteo.get(0, 0)}  "
                f"salto:{conteo.get(1, 0)}  "
                f"mantener:{conteo.get(2, 0)}  "
                f"pulso:{conteo.get(3, 0)}"
            )
            txt_datos = self.fuente_chica.render(info_datos, True, self.GRIS)
            self.pantalla.blit(txt_datos, (10, self.h - self.fuente_chica.get_linesize() - 8))

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
                        self.agachado = False
                        self.tecla_abajo_presionada = False
                        self._reset_estado_juego()
                        self.mostrar_menu()

                    elif e.key == pygame.K_f:
                        self._toggle_fullscreen()

                    elif e.key == pygame.K_SPACE and not self.modo_auto and self.en_suelo:
                        self.iniciar_salto()

                    elif e.key == pygame.K_DOWN and not self.modo_auto:
                        self.tecla_abajo_presionada = True
                        self.iniciar_agacharse()

                elif e.type == pygame.KEYUP:
                    if e.key == pygame.K_DOWN and not self.modo_auto:
                        self.tecla_abajo_presionada = False
                        self.terminar_agacharse()

            if not self.corriendo:
                break

            if self.modo_auto:
                accion = self.decision_auto()

                if accion == 1:
                    self.auto_agachado_timer = 0
                    self.auto_agachado_cooldown = 0
                    self.terminar_agacharse()
                    self.iniciar_salto()
                else:
                    self.actualizar_agachado_auto(accion)

            else:
                self.actualizar_agachado_manual_continuo()
                self.registrar_decision_manual()

            if self.salto:
                self.manejar_salto()

            self.actualizar_contadores_accion()

            if not self.bala_disparada:
                self.disparar_bala()

            if self.bala_disparada:
                self.frames_bala_actual += 1

            self._update_frame()
            pygame.display.flip()
            reloj.tick(45)

        pygame.quit()


def main() -> None:
    Juego().loop()


if __name__ == "__main__":
    main()
