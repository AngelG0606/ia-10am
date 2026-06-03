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

# Opcional: para graficar los datos en 2D y 3D
import matplotlib
# Configuramos backend para ventanas interactivas (TkAgg funciona en la mayoría de sistemas)
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass  # Usa el backend por defecto
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, necesario para activar 3D en matplotlib

# Activamos modo interactivo para que las ventanas no bloqueen el juego
plt.ion()


# Ventana base y factor de escala
BASE_W, BASE_H = 1080, 720
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1


@dataclass
class Sample:
    velocidad_bala: float
    distancia: float       # distancia firmada bala-jugador: >0 viene de frente, <0 ya paso
    altura_bala: float   # altura relativa de la bala (0.0=suelo, 1.0=nivel medio-cuerpo)
    accion: int  # 0=nada, 1=saltar, 2=agacharse


class Juego:
    def __init__(self) -> None:
        pygame.init()

        # Ventana fija (sin redimensionamiento automático) para evitar
        # problemas en pantallas muy grandes / 2K / 4K.
        self._flags = 0
        self._fullscreen = False

        # Tamaño fijo de ventana
        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + MLP (solo memoria)")

        # Colores
        self.BLANCO = (255, 255, 255)
        self.NEGRO = (0, 0, 0)
        self.GRIS = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)

        # Estado global
        self.corriendo = True
        self.modo_auto = False

        # Datos / modelo
        self.datos_modelo: List[Sample] = []
        self.modelo: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.modelo_entrenado = False
        # Caso especial: cuando solo hay una clase en los datos
        # (0 = nunca salto, 1 = siempre salto).
        self.clase_unica: Optional[int] = None
        # Debug / info del modelo en tiempo real
        self.ultima_proba_salto: Optional[float] = None
        self.ultima_proba_agacharse: Optional[float] = None

        # Parámetros de decisión
        self.decision_window = 500
        self.decision_record_every = 3
        self._decision_frame_counter = 0

        # Geometría / física (se rellenan en _apply_resolution)
        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size = (32, 48)
        self.bullet_size = (16, 16)
        self.ship_size = (64, 64)
        # Velocidad de desplazamiento del fondo
        self.fondo_speed = 3

        self.salto = False
        self.agachado = False
        self.en_suelo = True
        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0
        self.salto_vel = self.salto_vel_inicial
        # Timer para agachado automático (0 = sin timer / agachado manual indefinido)
        self.crouch_timer = 0
        self.crouch_duration = 8   # ~0.18 s a 45 fps — corto para permitir múltiples agachadas

        self.current_frame = 0
        self.frame_speed = 10
        self.frame_count = 0

        # Velocidad base de la bala (en píxeles/frame, negativa porque va de der→izq)
        self.velocidad_bala = -12
        self.bala_disparada = False
        # Altura relativa de la bala: 0.0 = nivel suelo, 1.0 = nivel medio-cuerpo
        self.altura_bala_relativa = 0.0
        self.fondo_x1 = 0
        self.fondo_x2 = start_w

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
        self.player_size_agachado = (int(32 * self.scale), int(24 * self.scale))  # mitad de altura
        self.bullet_size = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size = (int(64 * self.scale), int(64 * self.scale))
        self.fondo_speed = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 15 * self.scale
        self.gravedad = 1 * self.scale
        self.salto_vel = self.salto_vel_inicial
        # Altura media del cuerpo del jugador (para que la bala apunte aquí)
        self.bala_altura_media = int(self.player_size[1] // 2)

        self.decision_window = int(500 * self.scale)

        self.fuente = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(self.margin, self.ground_y, self.player_size[0], self.player_size[1])
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
            # Volver a ventana fija BASE_W x BASE_H
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    # ----------------- estado juego / modelo -----------------
    def _reset_estado_juego(self) -> None:
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.jugador.height = self.player_size[1]
        self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
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
        self._decision_frame_counter = 0
        self.fondo_x1 = 0
        self.fondo_x2 = self.w

    def _reset_modelo(self) -> None:
        self.modelo = None
        self.scaler = None
        self.modelo_entrenado = False
        self.clase_unica = None

    # ----------------- export / gráficas -----------------

    def exportar_datos_csv(self) -> str:
        """
        Exporta el contenido de self.datos_modelo a un CSV sencillo.
        Devuelve un mensaje con la ruta del archivo o el motivo del fallo.
        """
        if not self.datos_modelo:
            return "No hay datos para exportar."

        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")

        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["velocidad_bala", "distancia", "altura_bala", "accion"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.altura_bala, s.accion])
        except Exception as e:
            return f"Error al guardar CSV: {e}"

        return f"CSV guardado en datos_mlp.csv ({len(self.datos_modelo)} filas)."

    def graficar_datos_2d(self) -> str:
        """
        Grafica velocidad_bala vs distancia en 2D,
        coloreando por salto (0 / 1).
        Abre una ventana interactiva (desde el hilo principal, no bloqueante).
        """
        if not self.datos_modelo:
            return "No hay datos para graficar."

        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        color_map = {0: "blue", 1: "red", 2: "green"}
        cs = [color_map.get(s.accion, "gray") for s in self.datos_modelo]

        # Cerrar figura anterior si existe para evitar acumulación
        fig_num = plt.figure("Datos MLP - 2D", figsize=(8, 6)).number
        plt.figure(fig_num)
        plt.clf()

        ax = plt.gca()
        ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia firmada bala-jugador")
        ax.set_ylabel("Velocidad bala")
        ax.set_title("Datos MLP (azul=nada, rojo=salto, verde=agacha)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()

        return "Mostrando gráfica 2D interactiva (puedes rotar/zoom)."

    def graficar_datos_3d(self) -> str:
        """
        Grafica velocidad_bala vs distancia vs índice de tiempo (frame) en 3D,
        coloreando por salto (0 / 1).
        Abre una ventana interactiva (desde el hilo principal, no bloqueante).
        """
        if not self.datos_modelo:
            return "No hay datos para graficar."

        xs = [s.distancia for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        zs = [s.altura_bala for s in self.datos_modelo]
        color_map = {0: "blue", 1: "red", 2: "green"}
        cs = [color_map.get(s.accion, "gray") for s in self.datos_modelo]

        # Cerrar figura anterior si existe para evitar acumulación
        fig = plt.figure("Datos MLP - 3D", figsize=(8, 6))
        plt.clf()

        # Crear eje 3D correctamente desde la figura
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia firmada")
        ax.set_ylabel("Velocidad bala")
        ax.set_zlabel("Altura bala")
        ax.set_title("Datos MLP 3D (azul=nada, rojo=salto, verde=agacha)")
        plt.tight_layout()
        plt.show(block=False)
        plt.draw()

        return "Mostrando gráfica 3D interactiva (puedes rotar/zoom)."

    # ----------------- bala / salto / agacharse -----------------
    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            # Velocidad aleatoria
            self.velocidad_bala = int(random.randint(-12, -6) * self.scale)
            # Altura aleatoria: 0 = suelo (hay que saltar), 1 = nivel medio (hay que agacharse)
            self.altura_bala_relativa = random.choice([0.0, 1.0])
            if self.altura_bala_relativa == 0.0:
                # Bala a ras del suelo
                self.bala.y = self.ground_y + int(10 * self.scale)
            else:
                # Bala a mitad de cuerpo (nivel medio — hay que agacharse)
                self.bala.y = self.ground_y - self.bala_altura_media + int(14 * self.scale)
            self.bala_disparada = True

    def reset_bala(self) -> None:
        self.bala.x = self.w - self.margin
        self.bala_disparada = False

    def iniciar_salto(self) -> None:
        if self.en_suelo and not self.agachado:
            self.salto = True
            self.en_suelo = False

    def iniciar_agacharse(self) -> None:
        """Agachado MANUAL: dura hasta que el jugador suelte la tecla (↓).
        timer=0 significa sin límite de tiempo."""
        if self.en_suelo and not self.salto:
            if not self.agachado:
                self.agachado = True
                self.crouch_timer = 0  # sin timer: permanece hasta KEYUP
                self.jugador.height = self.player_size_agachado[1]
                self.jugador.y = self.ground_y + (self.player_size[1] - self.player_size_agachado[1])

    def iniciar_agacharse_auto(self) -> None:
        """Agachado AUTOMÁTICO para BALA BAJA: ciclos cortos y distintos (agacha→levanta→agacha),
        igual a presionar ↓ múltiples veces. Solo inicia cuando la agachada anterior ya terminó."""
        if self.en_suelo and not self.salto and not self.agachado:
            self.agachado = True
            self.crouch_timer = self.crouch_duration
            self.jugador.height = self.player_size_agachado[1]
            self.jugador.y = self.ground_y + (self.player_size[1] - self.player_size_agachado[1])

    def iniciar_agacharse_auto_sostenido(self) -> None:
        """Agachado AUTOMÁTICO para BALA ALTA: el personaje se queda agachado mientras el modelo
        prediga accion=2, igual a mantener presionada la flecha ↓. Reinicia el timer cada frame."""
        if self.en_suelo and not self.salto:
            if not self.agachado:
                self.agachado = True
                self.jugador.height = self.player_size_agachado[1]
                self.jugador.y = self.ground_y + (self.player_size[1] - self.player_size_agachado[1])
            # Reiniciar timer cada frame = mantenerse agachado continuamente
            self.crouch_timer = self.crouch_duration

    def terminar_agacharse(self) -> None:
        """Desactiva el estado agachado y resetea el timer."""
        if self.agachado:
            self.agachado = False
            self.crouch_timer = 0
            self.jugador.height = self.player_size[1]
            self.jugador.y = self.ground_y

    def manejar_agacharse(self) -> None:
        """Descuenta el timer de agachado automático.
        Solo actúa cuando timer > 0 (agachado manual con timer=0 se ignora: dura hasta KEYUP)."""
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

    # ----------------- datos / ML -----------------
    def distancia_bala_jugador(self) -> float:
        """Distancia horizontal firmada: positiva si la bala viene de frente."""
        return float(self.bala.x - self.jugador.x)

    def bala_alta_sobre_jugador(self) -> bool:
        """True mientras una bala alta aun esta cruzando por encima del jugador."""
        if self.altura_bala_relativa != 1.0:
            return False
        margen = int(4 * self.scale)
        return self.bala.right >= self.jugador.left - margen and self.bala.left <= self.jugador.right + margen

    def registrar_decision_manual(self) -> None:
        """Registra la acción del jugador en el frame actual.
        accion: 0=nada, 1=saltando, 2=agachado
        """
        if not self.bala_disparada:
            return
            
        # Evitar registrar datos basura cuando la bala ya pasó al jugador.
        # Esto previene que el modelo asocie "distancia negativa" con "saltar".
        if self.bala.right < self.jugador.left:
            return

        distancia = self.distancia_bala_jugador()
        if not self.en_suelo and self.salto:
            accion = 1
        elif self.agachado:
            accion = 2
        else:
            accion = 0
        self.datos_modelo.append(
            Sample(
                velocidad_bala=float(self.velocidad_bala),
                distancia=float(distancia),
                altura_bala=float(self.altura_bala_relativa),
                accion=accion,
            )
        )

    def balancear_samples(self, samples: List[Sample]) -> List[Sample]:
        """Duplica clases minoritarias para que el MLP no ignore saltar/agacharse."""
        por_clase = {clase: [s for s in samples if s.accion == clase] for clase in set(s.accion for s in samples)}
        max_count = max(len(grupo) for grupo in por_clase.values())
        rng = random.Random(42)
        balanceados: List[Sample] = []
        for clase in sorted(por_clase):
            grupo = por_clase[clase]
            balanceados.extend(grupo)
            faltan = max_count - len(grupo)
            if faltan > 0:
                balanceados.extend(rng.choice(grupo) for _ in range(faltan))
        rng.shuffle(balanceados)
        return balanceados

    def entrenar_modelo(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas más datos (>= 80). Juega en MANUAL."
        # Features: velocidad_bala, distancia, altura_bala
        X = [[s.velocidad_bala, s.distancia, s.altura_bala] for s in samples]
        y = [s.accion for s in samples]
        clases = sorted(set(y))
        # Si solo hay una clase, entrenamos un "modelo trivial"
        if len(clases) < 2:
            self._reset_modelo()
            self.clase_unica = int(clases[0])
            self.modelo_entrenado = True
            nombres = {0: "NADA", 1: "SIEMPRE SALTA", 2: "SIEMPRE AGACHA"}
            return True, f"Modelo trivial: {nombres.get(self.clase_unica, '?')}. Varía tus acciones para mejor modelo."
        conteo_original = Counter(y)
        samples = self.balancear_samples(samples)
        X = [[s.velocidad_bala, s.distancia, s.altura_bala] for s in samples]
        y = [s.accion for s in samples]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
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
        return True, f"MLP entrenado. Accuracy test ≈ {acc:.3f} | datos: {dict(conteo_original)}"

    def decision_auto(self) -> int:
        """Retorna la acción predicha: 0=nada, 1=saltar, 2=agacharse."""
        if not self.modelo_entrenado:
            return 0
        if not self.bala_disparada:
            return 0
            
        # Si la bala ya nos pasó completamente, la ignoramos y no hacemos nada
        if self.bala.right < self.jugador.left:
            return 0
            
        distancia = self.distancia_bala_jugador()

        # Caso especial: modelo trivial de una sola clase
        if self.clase_unica is not None and self.modelo is None:
            self.ultima_proba_salto = 1.0 if self.clase_unica == 1 else 0.0
            self.ultima_proba_agacharse = 1.0 if self.clase_unica == 2 else 0.0
            return int(self.clase_unica)

        # Caso normal: modelo MLP con scaler
        if self.modelo is None or self.scaler is None:
            return 0

        X = [[float(self.velocidad_bala), float(distancia), float(self.altura_bala_relativa)]]
        Xs = self.scaler.transform(X)
        accion = int(self.modelo.predict(Xs)[0])
        # Guardamos probabilidad de salto para info en pantalla
        if hasattr(self.modelo, "predict_proba"):
            probas = self.modelo.predict_proba(Xs)[0]
            clases = list(self.modelo.classes_)
            self.ultima_proba_salto = float(probas[clases.index(1)]) if 1 in clases else 0.0
            self.ultima_proba_agacharse = float(probas[clases.index(2)]) if 2 in clases else 0.0
        else:
            self.ultima_proba_salto = 1.0 if accion == 1 else 0.0
            self.ultima_proba_agacharse = 1.0 if accion == 2 else 0.0
        if self.agachado and self.bala_alta_sobre_jugador():
            return 2
        return accion

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
            f"Resolución: {self.w}x{self.h} | scale≈{self.scale:.2f} | ventana_decisión≈{self.decision_window}",
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
                # Ya no reaccionamos a cambios de tamaño de ventana,
                # la ventana es fija.
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

        # Dibujamos el jugador: si está agachado, escalamos el sprite a mitad de altura
        jugador_sprite = self.jugador_frames[self.current_frame]
        if self.agachado:
            sprite_agachado = pygame.transform.scale(
                jugador_sprite, self.player_size_agachado
            )
            self.pantalla.blit(sprite_agachado, (self.jugador.x, self.jugador.y))
        else:
            self.pantalla.blit(jugador_sprite, (self.jugador.x, self.jugador.y))

        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        if self.bala_disparada:
            self.bala.x += self.velocidad_bala
        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()
            # Al resetear también terminamos de agacharse en modo auto
            if self.modo_auto and self.agachado:
                self.terminar_agacharse()
        self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))

        # Si hay colisión, solo reiniciamos el estado del juego
        if self.jugador.colliderect(self.bala):
            self._reset_estado_juego()

        # HUD: info del estado actual
        y_hud = 10
        if self.agachado:
            txt_agachado = self.fuente_chica.render("[AGACHADO]", True, (100, 200, 255))
            self.pantalla.blit(txt_agachado, (10, y_hud))
            y_hud += self.fuente_chica.get_linesize() + 2
        # Indicador de altura de bala
        if self.bala_disparada:
            tipo_bala = "Bala ALTA (agáchate)" if self.altura_bala_relativa == 1.0 else "Bala BAJA (salta)"
            color_tip = (255, 120, 80) if self.altura_bala_relativa == 1.0 else (80, 220, 120)
            txt_bala = self.fuente_chica.render(tipo_bala, True, color_tip)
            self.pantalla.blit(txt_bala, (10, y_hud))
            y_hud += self.fuente_chica.get_linesize() + 2
        # Info del modelo en tiempo real (solo si hay modelo entrenado)
        if self.modelo_entrenado and self.modo_auto and self.ultima_proba_salto is not None:
            txt = self.fuente_chica.render(
                f"proba_salto≈{self.ultima_proba_salto:.2f}", True, self.AMARILLO
            )
            self.pantalla.blit(txt, (10, y_hud))

    def loop(self) -> None:
        reloj = pygame.time.Clock()
        self.mostrar_menu()

        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                # La ventana es de tamaño fijo: ignoramos eventos VIDEORESIZE.
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
                    if self.altura_bala_relativa == 1.0:
                        # Bala ALTA: se queda agachado mientras pasa la bala por encima
                        # (replica mantener ↓ presionado, sin ciclos de agacha→levanta)
                        if self.bala_alta_sobre_jugador():
                            self.iniciar_agacharse_auto_sostenido()
                        else:
                            self.iniciar_agacharse_auto()
                    else:
                        # Bala BAJA: ciclos cortos agacha→levanta
                        # (replica presionar ↓ varias veces)
                        self.iniciar_agacharse_auto()
                # El timer de manejar_agacharse() se encarga de terminar el agachado automáticamente
            else:
                # En modo manual registramos la decisión de este frame.
                self.registrar_decision_manual()

            if self.salto:
                self.manejar_salto()

            # Gestionar timer de agachado auto (inocuo si timer=0 o no está agachado)
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
