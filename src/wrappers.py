import gymnasium as gym
import numpy as np

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, shaping_type: str = "baseline", gamma: float = 1.0, 
                 step_cost_c: float = -0.01, potential_beta: float = 1.0, 
                 custom_c: float = 0.5, decay_rate: float = 1.0):
        super().__init__(env)
        self.shaping_type = shaping_type
        self.gamma = gamma
        self.step_cost_c = step_cost_c
        self.potential_beta = potential_beta
        self.custom_c = custom_c
        self.decay_rate = decay_rate
        
        self.last_state = None
        self.episode_count = 0
        
        # Access underlying environment details
        if hasattr(env.unwrapped, "desc"):
            self.desc = env.unwrapped.desc
            self.nrow, self.ncol = self.desc.shape
            self.n_states = self.nrow * self.ncol
        else:
            self.n_states = env.observation_space.n
            self.nrow = int(np.sqrt(self.n_states))
            self.ncol = self.nrow
        
        self.state_visits = np.zeros(self.n_states)
        self._find_goal_pos()
        
        # --- OPTIMIZATION: Pre-calculate Potentials ---
        self.potentials = np.zeros(self.n_states)
        # מחשבים פוטנציאלים גם עבור ה-Advanced
        if self.shaping_type in ["potential", "custom_advanced"]:
            for s in range(self.n_states):
                self.potentials[s] = self._calculate_potential_logic(s)

    def _find_goal_pos(self):
        if hasattr(self, 'desc'):
            goals = np.argwhere(self.desc == b'G')
            self.goal_pos = tuple(goals[0]) if len(goals) > 0 else (self.nrow - 1, self.ncol - 1)
        else:
            self.goal_pos = (self.nrow - 1, self.ncol - 1)

    def _get_distance(self, state: int) -> int:
        row, col = state // self.ncol, state % self.ncol
        g_row, g_col = self.goal_pos
        return abs(row - g_row) + abs(col - g_col)

    def _calculate_potential_logic(self, state: int) -> float:
        return -float(self._get_distance(state))

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_state = obs
        self.state_visits[obs] += 1
        self.episode_count += 1
        return obs, info

    def step(self, action: int):
        next_state, reward, done, truncated, info = super().step(action)
        
        if info is None: info = {}
        info['original_reward'] = reward

        self.state_visits[next_state] += 1
        
        F = 0.0
        
        # --- Baseline ---
        if self.shaping_type == "baseline":
            F = 0.0
            
        # --- Step Cost ---
        elif self.shaping_type == "step_cost":
            F = self.step_cost_c
            
        # --- Potential Based ---
        elif self.shaping_type == "potential":
            phi_next = self.potentials[next_state]
            phi_prev = self.potentials[self.last_state] if self.last_state is not None else 0.0
            F = self.potential_beta * ((self.gamma * phi_next) - phi_prev)
            
        # --- Custom 1: Safety First (בטיחות) ---
        elif self.shaping_type == "custom_safety":
            # הרעיון: עונש כבד מאוד אך ורק אם נופלים לבור.
            # אנחנו מוודאים שזה בור (done=True וגם reward=0) וגם שזה לא נגמר בגלל הגבלת צעדים (truncated)
            if done and reward == 0 and not truncated:
                F = -1.0  # עונש משמעותי על נפילה
            else:
                F = 0.0

        # --- Custom 2: Advanced Guidance (המשולב) ---
        elif self.shaping_type == "custom_advanced":
            # 1. Potential Component (המצפן)
            phi_next = self.potentials[next_state]
            phi_prev = self.potentials[self.last_state] if self.last_state is not None else 0.0
            potential_reward = self.potential_beta * ((self.gamma * phi_next) - phi_prev)
            
            # 2. Step Cost (יעילות)
            step_penalty = -0.001 
            
            # 3. Safety (עונש על בור)
            hole_penalty = 0.0
            if done and reward == 0 and not truncated:
                hole_penalty = -0.5 
            
            F = potential_reward + step_penalty + hole_penalty

        # Apply decay
        if self.decay_rate < 1.0:
            k = max(1, self.episode_count)
            decay_factor = self.decay_rate ** (k - 1)
            shaped_reward = reward + (F * decay_factor)
        else:
            shaped_reward = reward + F
        
        self.last_state = next_state
        return next_state, shaped_reward, done, truncated, info