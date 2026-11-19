"""Keyboard-to-velocity model for the teleoperation TUI.

The module deliberately knows nothing about curses or the drone SDK: it turns a
key name into a short velocity pulse and decays that pulse when the key is
released. That keeps the control law unit-testable on a laptop without hardware.

Axes follow Pioneer SDK 2 body-fixed convention: ``vx`` forward, ``vy`` left,
``vz`` up, ``yaw_rate`` positive counter-clockwise (degrees per second here,
radians per second inside the SDK).
"""

from __future__ import annotations

from dataclasses import dataclass, field

FORWARD = "forward"
BACK = "back"
LEFT = "left"
RIGHT = "right"
UP = "up"
DOWN = "down"
YAW_LEFT = "yaw_left"
YAW_RIGHT = "yaw_right"
STOP = "stop"

# Curses returns these names from ``getkey()``; plain letters are lowercase.
KEY_COMMANDS: dict[str, str] = {
    "w": FORWARD,
    "KEY_UP": FORWARD,
    "s": BACK,
    "KEY_DOWN": BACK,
    "a": LEFT,
    "KEY_LEFT": LEFT,
    "d": RIGHT,
    "KEY_RIGHT": RIGHT,
    "r": UP,
    "f": DOWN,
    "q": YAW_LEFT,
    "e": YAW_RIGHT,
    " ": STOP,
}

# Human-readable help lines rendered by the TUI.
KEY_HELP: tuple[tuple[str, str], ...] = (
    ("W/S, ↑/↓", "вперёд / назад"),
    ("A/D, ←/→", "влево / вправо"),
    ("R/F", "вверх / вниз"),
    ("Q/E", "поворот влево / вправо"),
    ("Пробел", "стоп"),
    ("T / L", "взлёт / посадка"),
    ("P", "фото"),
    ("[ / ]", "наклон камеры на 10°"),
    ("1/2/3/4", "камера: вниз -90° / -60° / -30° / горизонт 0°"),
    ("+ / -", "скорость"),
    ("V", "окно с камерой"),
    ("H", "справка"),
    ("X", "выход с посадкой"),
)

MIN_SPEED = 0.1
MAX_SPEED = 1.5
MIN_YAW_RATE = 10.0
MAX_YAW_RATE = 90.0


@dataclass(frozen=True)
class Velocity:
    """Body-fixed velocity command in SI units plus degrees per second."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0

    def is_zero(self) -> bool:
        return not any((self.vx, self.vy, self.vz, self.yaw_rate))


@dataclass
class ControlState:
    """Turns key presses into decaying velocity pulses.

    Each axis remembers when its pulse expires. Holding a key works because
    terminals repeat key presses; releasing it lets the axis fall back to zero
    after ``pulse_s`` seconds, so a lost terminal never leaves the drone flying
    away on a stale command.
    """

    speed: float = 0.5
    yaw_rate: float = 45.0
    pulse_s: float = 0.35
    max_speed: float = MAX_SPEED
    _expires: dict[str, tuple[str, float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.speed = _clamp(self.speed, MIN_SPEED, self.max_speed)
        self.yaw_rate = _clamp(self.yaw_rate, MIN_YAW_RATE, MAX_YAW_RATE)
        self.pulse_s = max(0.05, float(self.pulse_s))

    def press(self, command: str, now: float) -> None:
        """Start (or refresh) the pulse belonging to ``command``."""
        if command == STOP:
            self._expires.clear()
            return
        axis = _AXIS_OF.get(command)
        if axis is None:
            raise ValueError(f"unknown command: {command}")
        self._expires[axis] = (command, now + self.pulse_s)

    def tick(self, now: float) -> Velocity:
        """Drop expired axes and return the current velocity command."""
        for axis, (_, expires_at) in list(self._expires.items()):
            if now >= expires_at:
                del self._expires[axis]
        active = {command for command, _ in self._expires.values()}
        return Velocity(
            vx=self.speed if FORWARD in active else -self.speed if BACK in active else 0.0,
            vy=self.speed if LEFT in active else -self.speed if RIGHT in active else 0.0,
            vz=self.speed if UP in active else -self.speed if DOWN in active else 0.0,
            yaw_rate=(
                self.yaw_rate
                if YAW_LEFT in active
                else -self.yaw_rate if YAW_RIGHT in active else 0.0
            ),
        )

    def change_speed(self, delta: float) -> float:
        """Adjust and return the horizontal speed, clamped to safe limits."""
        self.speed = _clamp(self.speed + delta, MIN_SPEED, self.max_speed)
        return self.speed

    def stop(self) -> None:
        self._expires.clear()


# Command -> the axis it drives. Used to expire pulses independently.
_AXIS_OF: dict[str, str] = {
    FORWARD: "vx",
    BACK: "vx",
    LEFT: "vy",
    RIGHT: "vy",
    UP: "vz",
    DOWN: "vz",
    YAW_LEFT: "yaw",
    YAW_RIGHT: "yaw",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
