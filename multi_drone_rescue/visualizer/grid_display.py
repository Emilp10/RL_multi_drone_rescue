import pygame
from typing import Tuple, Dict
import numpy as np

# Colors
COLORS = {
    "empty": (230, 230, 230),
    "obstacle": (200, 60, 60),
    "victim": (250, 210, 70),
    "drone": (70, 120, 230),
    "grid": (180, 180, 180),
    "panel_bg": (245, 245, 250),
    "text": (30, 30, 30),
    "button": (220, 220, 235),
    "button_border": (100, 100, 120),
}


class GridDisplay:
    def __init__(self, grid_size: int, cell_size: int = 40, panel_width: int = 240):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.panel_width = panel_width
        self.grid_pix = grid_size * cell_size
        self.width = self.grid_pix + panel_width
        self.height = self.grid_pix
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Multi-Drone Rescue")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

        # UI elements (rects)
        self.reset_btn = pygame.Rect(self.grid_pix + 20, 20, self.panel_width - 40, 36)

    def draw(self, obstacles: np.ndarray, victims: np.ndarray, agents_pos: Dict[str, Tuple[int, int]], fps: int = 8, stats: Dict[str, float] | None = None):
        self.screen.fill((255, 255, 255))
        # grid area
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                color = COLORS["empty"]
                if obstacles[x, y] == 1:
                    color = COLORS["obstacle"]
                elif victims[x, y] == 1:
                    color = COLORS["victim"]
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, COLORS["grid"], rect, 1)

        # agents
        for i, (aid, (ax, ay)) in enumerate(agents_pos.items()):
            rect = pygame.Rect(ay * self.cell_size + 4, ax * self.cell_size + 4, self.cell_size - 8, self.cell_size - 8)
            pygame.draw.rect(self.screen, COLORS["drone"], rect, border_radius=6)

        # side panel
        panel_rect = pygame.Rect(self.grid_pix, 0, self.panel_width, self.height)
        pygame.draw.rect(self.screen, COLORS["panel_bg"], panel_rect)

        # Reset button
        pygame.draw.rect(self.screen, COLORS["button"], self.reset_btn, border_radius=6)
        pygame.draw.rect(self.screen, COLORS["button_border"], self.reset_btn, 1, border_radius=6)
        self._blit_text("Reset (R)", self.reset_btn.centerx, self.reset_btn.centery, center=True)

        # Legend and stats
        y0 = 80
        self._blit_text("Legend:", self.grid_pix + 20, y0)
        self._legend_swatch(self.grid_pix + 20, y0 + 20, COLORS["drone"], "Drone")
        self._legend_swatch(self.grid_pix + 20, y0 + 45, COLORS["victim"], "Victim (click to add/remove)")
        self._legend_swatch(self.grid_pix + 20, y0 + 70, COLORS["obstacle"], "Obstacle (click to toggle)")

        y1 = y0 + 110
        self._blit_text("Stats:", self.grid_pix + 20, y1)
        if stats:
            self._blit_text(f"Episode: {int(stats.get('episode', 0))}", self.grid_pix + 20, y1 + 20)
            self._blit_text(f"Step: {int(stats.get('step', 0))}", self.grid_pix + 20, y1 + 40)
            self._blit_text(f"Ep Return: {stats.get('ep_return', 0.0):.1f}", self.grid_pix + 20, y1 + 60)
            self._blit_text(f"Victims left: {int(stats.get('victims_left', 0))}", self.grid_pix + 20, y1 + 80)
            mode = stats.get('mode')
            if mode:
                self._blit_text(f"Mode: {mode}  (G to toggle)", self.grid_pix + 20, y1 + 100)

        pygame.display.flip()
        self.clock.tick(fps)

    def _blit_text(self, text: str, x: int, y: int, center: bool = False):
        surf = self.font.render(text, True, COLORS["text"])
        rect = surf.get_rect()
        if center:
            rect.center = (x, y)
        else:
            rect.topleft = (x, y)
        self.screen.blit(surf, rect)

    def _legend_swatch(self, x: int, y: int, color: Tuple[int, int, int], label: str):
        rect = pygame.Rect(x, y, 18, 18)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, COLORS["grid"], rect, 1)
        self._blit_text(label, x + 26, y + 2)

    def handle_click(self, pos: Tuple[int, int]):
        x = pos[1] // self.cell_size
        y = pos[0] // self.cell_size
        return int(x), int(y)

    def click_in_grid(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.grid_pix and 0 <= pos[1] < self.grid_pix

    def clicked_reset(self, pos: Tuple[int, int]) -> bool:
        return self.reset_btn.collidepoint(pos)

    def close(self):
        pygame.quit()
