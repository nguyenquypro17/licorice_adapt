import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
import random
import torch
import cv2
import wandb

from gymnasium.envs.registration import register
from custom_dynamic_obstacles_env import CustomDynamicObstaclesEnv
from minigrid.core.constants import DIR_TO_VEC, COLOR_TO_IDX, OBJECT_TO_IDX

def set_random_seed(seed):
    assert torch.cuda.is_available()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class DynamicObstaclesEnvWrapper(gym.Wrapper):
    """Wraps the DynamicObstacles environment."""
    def __init__(self, env, ROWS, COLS, concept_version=1):
        super(DynamicObstaclesEnvWrapper, self).__init__(env)
        self.env = env
        self.ROWS = ROWS
        self.COLS = COLS
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, self.ROWS, self.COLS), dtype=np.uint8)
        self.concept_version = concept_version
        if concept_version == 1:
            self.task_types = ['classification'] * 5
            self.num_classes = [2, 2, 4, 2, 2]
            self.concept_names = ['agent_at_right', 'agent_at_bottom', 'agent_direction', 'obstacle_right', 'obstacle_below']
        elif concept_version == 3:
            if self.env.unwrapped.grid.width == 5:
                self.task_types = ['classification'] * 11
                self.num_classes = [6, 6, 4, 6, 6, 6, 6, 2, 2, 2, 2]
                self.concept_names = [
                    'agent_position_x', 'agent_position_y', 'agent_direction', 
                    'obstacle1_position_x', 'obstacle1_position_y', 
                    'obstacle2_position_x', 'obstacle2_position_y', 
                    'movable_right', 'movable_down', 'movable_left', 'movable_up'
                ]
        self.reset()

    def get_concept(self):
        if self.concept_version == 1:
            return self._get_concept_v1()
        elif self.concept_version == 3:
            return self._get_concept_v3()

    def _get_concept_v1(self):
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        grid = self.env.unwrapped.grid
        agent_at_right = (agent_pos[0] == grid.width-1 - 1)
        agent_at_bottom = (agent_pos[1] == grid.height-1 - 1)
        cell_right = grid.get(agent_pos[0] + 1, agent_pos[1]) if agent_pos[0] + 1 < grid.width-1 else None
        obstacle_right = (cell_right != None and cell_right.type == 'ball')
        cell_below = grid.get(agent_pos[0], agent_pos[1] + 1) if agent_pos[1] + 1 < grid.height-1 else None
        obstacle_below = (cell_below != None and cell_below.type == 'ball')
        numbers = [int(agent_at_right), int(agent_at_bottom), agent_dir, int(obstacle_right), int(obstacle_below)]
        return np.array(numbers, dtype=np.float32)

    def _get_concept_v3(self):
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        grid = self.env.unwrapped.grid
        def get_cell(x, y):
            if x < 0 or x >= grid.width or y < 0 or y >= grid.height:
                return None
            return grid.get(x, y)
        agent_position = (agent_pos[0], agent_pos[1])
        agent_direction = agent_dir
        obstacle_positions = []
        for x in range(grid.width):
            for y in range(grid.height):
                cell = get_cell(x, y)
                if cell and cell.type == 'ball':
                    obstacle_positions.append((x, y))
        obstacle_positions = sorted(obstacle_positions)
        assert len(obstacle_positions) == 2 or len(obstacle_positions) == 3
        direction_movable = {
            'right': self.can_move(agent_pos, 0, grid),
            'down': self.can_move(agent_pos, 1, grid),
            'left': self.can_move(agent_pos, 2, grid),
            'up': self.can_move(agent_pos, 3, grid),
        }
        numbers = [agent_position[0], agent_position[1], agent_direction]
        for i in range(len(obstacle_positions)):
            numbers.extend([obstacle_positions[i][0], obstacle_positions[i][1]])
        numbers.extend([int(direction_movable['right']), int(direction_movable['down']), int(direction_movable['left']), int(direction_movable['up'])])
        return np.array(numbers, dtype=np.float32)

    def can_move(self, position, direction, grid):
        next_pos = DIR_TO_VEC[direction]
        if 1 <= position[0] + next_pos[0] < grid.width-1 and 1 <= position[1] + next_pos[1] < grid.height-1:
            next_cell = grid.get(position[0] + next_pos[0], position[1] + next_pos[1])
            return next_cell is None or next_cell.can_overlap()
        else:
            return False

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        img = self.get_image()
        return img, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        img = self.get_image()
        if done:
            info['terminal_observation'] = img
        return img, reward, done, truncated, info

    def get_image(self):
        img = self.env.render()
        img = cv2.resize(img, (self.COLS, self.ROWS))
        img = img.transpose((2, 0, 1))
        return img

class DoorKeyEnvWrapper(gym.Wrapper):
    """Wraps the DoorKey environment."""
    def __init__(self, env, ROWS, COLS):
        super(DoorKeyEnvWrapper, self).__init__(env)
        self.env = env
        self.ROWS = ROWS
        self.COLS = COLS
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, self.ROWS, self.COLS), dtype=np.uint8)
        self.task_types = ['classification']*12
        self.num_classes = [8, 8, 4, 8, 8, 8, 8, 2, 2, 2, 2, 2]
        self.concept_names = ['agent_position_x', 'agent_position_y', 'agent_direction', 'key_position_x', 'key_position_y', 'door_position_x', 'door_position_y', 'door_open', 'direction_movable_right', 'direction_movable_down', 'direction_movable_left', 'direction_movable_up']
        self.reset()

    def get_concept(self):
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        grid = self.env.unwrapped.grid
        key_pos = (0, 0)
        door_pos = None
        door_open = False
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None:
                    if cell.type == 'door':
                        door_pos = (x, y)
                        door_open = cell.is_open
                    elif cell.type == 'key':
                        key_pos = (x, y)
        direction_movable = {
            'right': self.can_move(agent_pos, 0, grid),
            'down': self.can_move(agent_pos, 1, grid),
            'left': self.can_move(agent_pos, 2, grid),
            'up': self.can_move(agent_pos, 3, grid),
        }
        infos = {
            'agent_position': agent_pos,
            'agent_direction': agent_dir,
            'key_position': key_pos,
            'door_position': door_pos,
            'door_open': door_open,
            'direction_movable': direction_movable
        }
        numbers = []
        for key, value in infos.items():
            if key == 'direction_movable':
                for k, v in value.items():
                    numbers.append(int(v))
            elif isinstance(value, tuple):
                numbers.extend([x for x in value])
            else:
                numbers.append(value)
        return np.array(numbers, dtype=np.float32)

    def can_move(self, position, direction, grid):
        next_pos = DIR_TO_VEC[direction]
        if 1 <= position[0] + next_pos[0] < grid.width-1 and 1 <= position[1] + next_pos[1] < grid.height-1:
            next_cell = grid.get(position[0] + next_pos[0], position[1] + next_pos[1])
            return next_cell is None or next_cell.can_overlap()
        else:
            return False

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        img = self.get_image()
        return img, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        img = self.get_image()
        if done:
            info['terminal_observation'] = img
        return img, reward, done, truncated, info

    def get_image(self):
        img = self.env.render()
        img = cv2.resize(img, (self.COLS, self.ROWS))
        img = img.transpose((2, 0, 1))
        return img

def train_environment(game_name, grid_size, total_timesteps, seed):
    """Train a single environment configuration."""
    config = {
        "total_timesteps": total_timesteps,
        "num_envs": 8,
        "env_name": "Minigrid",
        "concept_loss_type": "vanilla_freeze",
        "con_coef": 0.5,
        "n_steps": 256 * 2,
        "n_epochs": 10,
        "batch_size": 128 * 2,
        "ent_coef": 0.01,
        "learning_rate": 3e-4,
        "vf_coef": 0.5,
        "share_features_extractor": True,
        "intervention": False,
        "game": game_name,
        "seed": seed,
        "concept_version": 3 if game_name == 'DynamicObstacles' else 1,
    }

    ROWS = 160
    COLS = 160

    set_random_seed(config["seed"])

    if game_name == 'DynamicObstacles':
        register(
            id=f'MiniGrid-Custom-Dynamic-Obstacles-{grid_size}x{grid_size}-v0',
            entry_point='custom_dynamic_obstacles_env:CustomDynamicObstaclesEnv',
            kwargs={'size': grid_size, 'n_obstacles': grid_size // 2}
        )
        env_id = f'MiniGrid-Custom-Dynamic-Obstacles-{grid_size}x{grid_size}-v0'
        env = make_vec_env(env_id, n_envs=config["num_envs"], seed=config["seed"], env_kwargs={'highlight': False}, wrapper_class=DynamicObstaclesEnvWrapper, wrapper_kwargs={"ROWS": ROWS, "COLS": COLS, "concept_version": config["concept_version"]})
        non_vectorized_env = gym.make(env_id, render_mode="rgb_array", highlight=False)
        non_vectorized_env = DynamicObstaclesEnvWrapper(non_vectorized_env, ROWS, COLS, config["concept_version"])
        non_vectorized_env.reset(seed=config["seed"])
        run_name = f"DynamicObstacles-{grid_size}x{grid_size}-seed{seed}"

    elif game_name == 'DoorKey':
        register(
            id=f"MiniGrid-DoorKey-{grid_size}x{grid_size}-v0",
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": grid_size},
        )
        env_id = f"MiniGrid-DoorKey-{grid_size}x{grid_size}-v0"
        env = make_vec_env(env_id, n_envs=config["num_envs"], seed=config["seed"], env_kwargs={'highlight': False}, wrapper_class=DoorKeyEnvWrapper, wrapper_kwargs={"ROWS": ROWS, "COLS": COLS})
        non_vectorized_env = gym.make(env_id, render_mode="rgb_array", highlight=False)
        non_vectorized_env = DoorKeyEnvWrapper(non_vectorized_env, ROWS, COLS)
        non_vectorized_env.reset(seed=config["seed"])
        run_name = f"DoorKey-{grid_size}x{grid_size}-seed{seed}"

    config["run_name"] = run_name

    run = wandb.init(
        project=f"concept-RL-Minigrid-{config['game']}",
        group=run_name,
        name=run_name,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    model = PPO(
        "CnnPolicy",
        env,
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        learning_rate=lambda progression: config["learning_rate"] * progression,
        ent_coef=config["ent_coef"],
        clip_range=lambda progression: 0.1 * progression,
        batch_size=config["batch_size"],
        verbose=1,
        seed=config["seed"],
        tensorboard_log="runs",
        policy_kwargs={
            "concept_dim": len(non_vectorized_env.get_concept()),
            "task_types": non_vectorized_env.task_types,
            "num_classes": non_vectorized_env.num_classes,
            "concept_names": non_vectorized_env.concept_names,
            "share_features_extractor": config["share_features_extractor"],
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
            "focal_loss": False,
        },
        concept_loss_type=config["concept_loss_type"],
        con_coef=config["con_coef"],
        intervention=config["intervention"],
        non_vectorized_env=non_vectorized_env,
        num_samples=500,
        accept_rate=1,
        active_learning=False,
        unlabeled_set_ratio=10,
        model_ensembles=5,
        gpt4o=False,
        hashing=True,
        gpt4o_prompt=None,
        gpt4o_checker=None,
        gpt4o_path=None,
    )

    eval_env = env
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=20,
        best_model_save_path=f"models/{run.id}",
        log_path=f"models/{run.id}",
        eval_freq=(config["total_timesteps"] // 10 // config["num_envs"] - 1),
        deterministic=True,
        render=False)
    cb_list = CallbackList([eval_callback, WandbCallback()])

    model.learn(
        total_timesteps=config["total_timesteps"],
        query_num_times=1,
        query_labels_per_time=500,
        callback=cb_list,
    )

    model.save(f"models/{run.id}/model.zip")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    wandb.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})

    env.close()
    wandb.finish()

    print(f"✓ Training complete for {game_name} {grid_size}x{grid_size}")

if __name__ == "__main__":
    SEED = 42
    EPISODES = 1000

    print("=" * 80)
    print("Training: Minigrid DoorKey 6x6 v0")
    print(f"Episodes: {EPISODES}, Seed: {SEED}")
    print("=" * 80)
    train_environment("DoorKey", 6, EPISODES, SEED)

    print("\n" + "=" * 80)
    print("Training: Minigrid Dynamic Obstacles 5x5 v0")
    print(f"Episodes: {EPISODES}, Seed: {SEED}")
    print("=" * 80)
    train_environment("DynamicObstacles", 5, EPISODES, SEED)

    print("\n" + "=" * 80)
    print("All training complete!")
    print("=" * 80)