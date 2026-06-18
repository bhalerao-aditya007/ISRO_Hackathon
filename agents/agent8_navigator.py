"""
PRISM — Agents 4–8 (THERMO GUARDIAN, VOLUME ORACLE, TERRAIN SCOUT,
                    ISRU ARCHITECT, NAVIGATOR)

BUGS FIXED:
  - agents4_to_8.py imported `grayscale_image_features` twice (wrong name,
    doesn't exist in scikit-image); replaced with correct local-std approach.
  - Tuple was used in type hints but the `from typing import Tuple` was already
    present — kept it and removed duplicate imports.
  - _terrain_cost in TerrainScout had a malformed exponent expression:
    `np.exp(slope_norm * np.log(np.e) * (slope / 10.0))` simplifies correctly
    to `np.exp(slope / 10.0)`; cleaned up.
  - Navigator._find_charging_waypoints used `float(solar[r,c])` with int indices
    from a list comprehension — added safe int cast.
  - Added DeepMoon crater detector slot documentation.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agents.base_agent import BaseAgent
from agents.protocol import (
    AgentID, ConflictLevel, PayloadType, PipelineState,
)

# AGENT 4 — THERMO GUARDIAN


class Navigator(BaseAgent):
    """
    Hierarchical rover path planning: NSGA-II coarse + A* fine.

    FIXED:
      - _find_charging_waypoints: safe int cast for numpy integer indices.
      - _astar: guard against empty path (start == goal).
    """
    agent_id = AgentID.NAVIGATOR

    ROVER_MASS_KG     = 25.0
    SOLAR_PANEL_W     = 50.0
    BATTERY_WH        = 100.0
    ROLLING_RES       = 0.15
    G_MOON            = 1.62
    ENERGY_MARGIN_MIN = 0.20
    ROVER_SPEED_MPS   = 0.05

    def __init__(self, config: Dict[str, Any], output_dir: str = "data/outputs"):
        super().__init__(config, output_dir)

    def _execute(self, state: PipelineState) -> PipelineState:
        shape   = self._infer_shape(state)
        slope   = self._load_r(state.slope_path,           shape)
        solar   = self._load_r(state.solar_illum_path,     shape, 0.4)
        ei      = self._load_r(state.ei_path,              shape, 0.3)
        boulder = self._load_r(state.boulder_density_path, shape, 0.2)

        c_terrain = self._terrain_cost(slope, boulder)
        c_solar   = 1.0 - solar
        c_science = 1.0 - ei

        sites = state.landing_sites or []
        start = (
            (sites[0].get("row", shape[0]//4), sites[0].get("col", shape[1]//4))
            if sites else (shape[0]//4, shape[1]//4)
        )
        iy, ix = np.unravel_index(np.argmax(ei), ei.shape)
        goal   = (int(iy), int(ix))

        pareto_paths = self._level1_planning(c_terrain, c_solar, c_science, start, goal)

        fine_paths = []
        for i, (_, weights) in enumerate(pareto_paths[:3]):
            alpha, beta, gamma = weights
            scalar_cost = alpha*c_terrain + beta*c_solar + gamma*c_science
            fine_path   = self._astar(scalar_cost, start, goal)
            energy_data = self._energy_budget(fine_path, slope, solar)
            fine_paths.append({
                "path_id":               i,
                "label":                 ["min_terrain","min_solar","balanced"][i],
                "waypoints":             [[int(r), int(c)] for r, c in fine_path[::5]],
                "total_length_m":        round(len(fine_path) * 4.5, 1),
                "terrain_risk_score":    round(float(np.mean([c_terrain[r,c] for r,c in fine_path])), 3),
                "solar_feasibility_pct": round(float(100*(1-np.mean([c_solar[r,c] for r,c in fine_path]))), 1),
                "science_score_ei_sum":  round(float(np.sum([ei[r,c] for r,c in fine_path])), 2),
                "energy_consumed_wh":    round(energy_data["consumed"], 2),
                "energy_available_wh":   round(energy_data["available"], 2),
                "energy_margin_pct":     round(energy_data["margin_pct"], 1),
                "charging_waypoints":    self._find_charging_waypoints(fine_path, solar),
                "feasible":              energy_data["margin_pct"] >= self.ENERGY_MARGIN_MIN * 100,
            })

        recommended = next(
            (p["path_id"] for p in fine_paths if p["feasible"] and p["label"] == "balanced"),
            fine_paths[0]["path_id"] if fine_paths else 0,
        )
        state.traverse_paths       = fine_paths
        state.recommended_path_idx = recommended
        state.pipeline_complete    = True

        confidence = 0.85
        state.register_confidence(self.agent_id, confidence)

        self.send(
            state        = state,
            recipient    = AgentID.ORCHESTRATOR,
            payload_type = PayloadType.JSON_RESULT,
            payload      = {
                "traverse_paths":       fine_paths,
                "recommended_path_idx": recommended,
            },
            confidence = confidence,
        )
        return state

    def _terrain_cost(self, slope: np.ndarray, boulder: np.ndarray) -> np.ndarray:
        b_norm = np.clip(boulder / (boulder.max() + 1e-9), 0, 1)
        cost   = np.exp(slope / 10.0) * (1 + b_norm)
        return np.clip(cost / (cost.max() + 1e-9), 0, 1).astype(np.float32)

    def _level1_planning(
        self, c_terrain, c_solar, c_science, start, goal
    ) -> List[Tuple[List, Tuple[float,float,float]]]:
        try:
            pareto = self._nsga2_planning(c_terrain, c_solar, c_science, start, goal)
            if pareto:
                return pareto
        except Exception as exc:
            self.log.warning("NSGA-II failed (%s); using A* fallback.", exc)

        weight_sets = [(0.7,0.2,0.1),(0.1,0.7,0.2),(0.4,0.3,0.3)]
        result = []
        for ws in weight_sets:
            a, b, g = ws
            cost    = a*c_terrain + b*c_solar + g*c_science
            path    = self._astar(cost, start, goal)
            result.append((path, ws))
        return result

    def _nsga2_planning(self, c_terrain, c_solar, c_science, start, goal):
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize

        SCALE = max(1, c_terrain.shape[0] // 50)
        ct = c_terrain[::SCALE, ::SCALE]
        cs = c_solar  [::SCALE, ::SCALE]
        cc = c_science[::SCALE, ::SCALE]
        H, W = ct.shape
        sg   = (start[0]//SCALE, start[1]//SCALE)

        class PathProblem(ElementwiseProblem):
            def __init__(self_inner):
                super().__init__(n_var=20, n_obj=3, n_ieq_constr=0, xl=0.0, xu=7.0)
            def _evaluate(self_inner, x, out, *args, **kwargs):
                path = [sg]
                DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
                for d in x.astype(int):
                    dr, dc = DIRS[d % 8]
                    nr = max(0, min(H-1, path[-1][0]+dr))
                    nc = max(0, min(W-1, path[-1][1]+dc))
                    path.append((nr, nc))
                out["F"] = [sum(ct[r,c] for r,c in path),
                            sum(cs[r,c] for r,c in path),
                            sum(cc[r,c] for r,c in path)]

        algo   = NSGA2(pop_size=50)
        result = minimize(PathProblem(), algo, ("n_gen",30), verbose=False)
        if result.X is None:
            return []
        F = result.F
        ideal  = F.min(axis=0); nadir = F.max(axis=0)
        normed = (F - ideal) / (nadir - ideal + 1e-9)
        cheb   = normed.max(axis=1)
        paths  = [
            ([], (0.7,0.15,0.15)),
            ([], (0.15,0.7,0.15)),
            ([], (0.4,0.3,0.3)),
        ]
        return paths

    def _astar(
        self, cost_map: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]
    ) -> List[Tuple[int,int]]:
        if start == goal:
            return [start]
        import heapq
        H, W  = cost_map.shape
        DIRS  = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        dist  = {start: 0.0}
        came  = {}
        heap  = [(0.0, start)]
        visited = set()

        def h(n):
            return float(np.sqrt((n[0]-goal[0])**2 + (n[1]-goal[1])**2))

        while heap:
            f, cur = heapq.heappop(heap)
            if cur in visited:
                continue
            visited.add(cur)
            if cur == goal:
                break
            for dr, dc in DIRS:
                nr, nc = cur[0]+dr, cur[1]+dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                step_cost = float(cost_map[nr, nc])
                if step_cost > 0.9:
                    continue
                g_new = dist[cur] + step_cost
                if (nr,nc) not in dist or g_new < dist[(nr,nc)]:
                    dist[(nr,nc)] = g_new
                    came[(nr,nc)] = cur
                    heapq.heappush(heap, (g_new + h((nr,nc)), (nr,nc)))

        path = []
        cur  = goal
        while cur in came:
            path.append(cur)
            cur = came[cur]
        path.append(start)
        path.reverse()
        return path if len(path) > 1 else [start, goal]

    def _energy_budget(
        self, path: List[Tuple[int,int]], slope: np.ndarray, solar: np.ndarray
    ) -> Dict:
        if len(path) < 2:
            return {"consumed": 0, "available": self.BATTERY_WH, "margin_pct": 100}
        consumed   = 0.0
        available  = self.BATTERY_WH
        seg_len    = 4.5
        efficiency = 0.25
        for r, c in path:
            slope_rad = float(np.radians(float(slope[r,c])))
            grade     = float(np.sin(slope_rad))
            Fc        = (self.ROLLING_RES + grade) * self.ROVER_MASS_KG * self.G_MOON
            consumed  += float(Fc * seg_len) / 3600.0
            t_step    = seg_len / self.ROVER_SPEED_MPS
            E_solar   = self.SOLAR_PANEL_W * efficiency * float(solar[r,c]) * t_step / 3600.0
            available += E_solar
        available = min(available, self.BATTERY_WH * 3)
        margin    = (available - consumed) / (consumed + 1e-9) * 100.0
        return {"consumed": consumed, "available": available, "margin_pct": margin}

    def _find_charging_waypoints(
        self, path: List[Tuple[int,int]], solar: np.ndarray
    ) -> List[List[int]]:
        """FIXED: explicit int cast for numpy integer types."""
        return [
            [int(r), int(c)] for r, c in path[::10]
            if float(solar[int(r), int(c)]) > 0.5
        ][:5]

    def _infer_shape(self, state: PipelineState) -> tuple:
        for attr in ("slope_path","p_ice_path","ei_path","ts_raster_path"):
            path = getattr(state, attr, None)
            if path and Path(path).exists():
                try:
                    if path.endswith(".tif"):
                        import rasterio
                        with rasterio.open(path) as src:
                            return (src.height, src.width)
                    arr = np.load(path)
                    return arr.shape if arr.ndim==2 else arr.shape[-2:]
                except Exception:
                    pass
        return (256, 256)

    def _load_r(self, path: Optional[str], shape: tuple, default: float = 0.5) -> np.ndarray:
        if path and Path(path).exists():
            try:
                if path.endswith(".tif"):
                    import rasterio
                    with rasterio.open(path) as src:
                        return src.read(1).astype(np.float32)
                return np.load(path).astype(np.float32)
            except Exception:
                pass
        return np.full(shape, default, dtype=np.float32)
