import gymnasium as gym
import numpy as np
import time
import sys
import os

# הוספת התיקייה הנוכחית ל-path כדי שהייבוא יעבוד חלק
sys.path.append(os.getcwd())

from src.maps import get_map
from src.sarsa_agent import SarsaAgent
from src.mc_agent import MonteCarloAgent
from src.wrappers import RewardShapingWrapper

def visualize_policy(agent_type="SARSA", shaping_type="custom_advanced"):
    """
    מאמן סוכן זריז ומציג את הביצועים שלו ויזואלית על המסך.
    """
    # הגדרות כמו בניסוי
    map_size = 6
    desc = get_map(map_size)
    
    # 1. יצירת סביבת אימון (בלי גרפיקה - כדי שיהיה מהיר)
    print(f"--- Training {agent_type} agent with {shaping_type} shaping... ---")
    train_env = gym.make("FrozenLake-v1", desc=desc, is_slippery=True)
    
    # עטיפה עם ה-Shaping שבחרנו
    train_env = RewardShapingWrapper(
        train_env, 
        shaping_type=shaping_type,
        step_cost_c=-0.01,
        potential_beta=1.0,
        custom_c=0.5
    )
    
    n_actions = train_env.action_space.n
    n_states = train_env.observation_space.n
    
    # אתחול הסוכן עם פרמטרים טובים (מהניסויים שלך)
    if agent_type == "SARSA":
        agent = SarsaAgent(n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=0.99)
    else:
        agent = MonteCarloAgent(n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=0.99)
        
    # אימון קצר יחסית (3000 פרקים מספיקים בדרך כלל כדי ללמוד את המפה הזו)
    for ep in range(3000):
        obs, _ = train_env.reset()
        done = False
        action = agent.get_action(obs)
        
        while not done:
            next_state, reward, done, truncated, _ = train_env.step(action)
            
            if agent_type == "SARSA":
                next_action = agent.get_action(next_state)
                agent.update(obs, action, reward, next_state, next_action, done)
                obs, action = next_state, next_action
            else:
                # MC Logic (פשוט יותר לדמו - רק אוסף מידע, העדכון האמיתי קורה בסוף אבל כאן אנחנו רק רוצים שהסוכן יזוז)
                # הערה: עבור דמו מהיר, SARSA עדיף כי הוא מתעדכן תוך כדי תנועה.
                # ב-MC מלא היינו צריכים לשמור את כל הפרק. כאן נשתמש ב-SARSA כברירת מחדל.
                obs = next_state
                action = agent.get_action(obs)
                
    print("Training finished. Starting visualization...")
    train_env.close()

    # 2. שלב ההצגה (עם גרפיקה)
    # render_mode='human' פותח את החלון
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=True, render_mode="human")
    env = RewardShapingWrapper(env, shaping_type=shaping_type) # רק בשביל התאימות
    
    state, _ = env.reset()
    done = False
    
    # מכבים את האקספלורציה (Epsilon=0) כדי לראות את המדיניות הכי טובה שנלמדה (Exploitation בלבד)
    agent.epsilon = 0.0
    
    print("Agent is running... (Look at the popup window)")
    
    steps = 0
    total_reward = 0
    
    while not done and steps < 100:
        # הצגת הפעולה
        action = agent.get_action(state)
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        # השהייה קטנה כדי שיהיה אפשר לראות את הצעדים בעין
        time.sleep(0.5) 
        
    env.close()
    
    if total_reward > 0:
        print(f"RESULT: Success! The agent reached the goal in {steps} steps.")
    else:
        print("RESULT: Failed. The agent fell into a hole or got stuck.")

if __name__ == "__main__":
    # הרצת הדמו עם הסוכן המנצח שלך
    visualize_policy(agent_type="SARSA", shaping_type="custom_advanced")