import numpy as np
import matplotlib.pyplot as plt
import imageio
import random
import networkx as nx

class RoomFloorEnv:
    """
    A 2D grid environment with rooms separated by walls (0=free, 1=wall).
    The agent moves only if there's no wall in the target cell.
    """
    def __init__(self, layout):
        self.grid = np.array(layout)
        self.rows, self.cols = self.grid.shape
        self.agent_pos = self.find_free_cell()

    def find_free_cell(self):
        """Return the first free cell (0) from top-left scanning."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 0:
                    return (r, c)
        return (0, 0)  # fallback if no free cell

    def reset(self):
        """Reset agent position to a free cell."""
        self.agent_pos = self.find_free_cell()
        return self.agent_pos

    def step(self, action):
        """
        0=up, 1=down, 2=left, 3=right
        Move agent if not blocked by wall or boundary.
        """
        r, c = self.agent_pos
        nr, nc = r, c
        if action == 0 and r > 0:
            nr = r - 1
        elif action == 1 and r < self.rows - 1:
            nr = r + 1
        elif action == 2 and c > 0:
            nc = c - 1
        elif action == 3 and c < self.cols - 1:
            nc = c + 1

        # Check for wall
        if self.grid[nr, nc] == 1:
            nr, nc = r, c  # remain in place

        self.agent_pos = (nr, nc)
        return self.agent_pos


class MockTDB:
    """
    Assign each free cell a random code in [1..K].
    In a real TDB, you'd compute the code from observations.
    """
    def __init__(self, layout, K=10, seed=0):
        np.random.seed(seed)
        self.layout = np.array(layout)
        self.rows, self.cols = self.layout.shape
        self.codes = -1 * np.ones((self.rows, self.cols), dtype=int)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.layout[r, c] == 0:
                    self.codes[r, c] = np.random.randint(1, K + 1)

    def get_code(self, r, c):
        return self.codes[r, c]


def draw_floor_plan(ax, layout, agent_pos=None):
    """
    Display the top-down floor plan (0=free -> white, 1=wall -> black),
    plus a red dot for the agent if agent_pos is given.
    """
    layout = np.array(layout)
    rows, cols = layout.shape
    ax.set_title("Floor Plan")
    img = np.zeros((rows, cols))
    img[layout == 1] = 1  # walls -> black
    ax.imshow(img, cmap='gray_r', origin='upper')

    # Grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='k', linestyle='-', linewidth=0.5)

    # Agent
    if agent_pos is not None:
        r, c = agent_pos
        ax.plot(c, r, 'ro', markersize=10)


def run_simulation(layout, num_steps=15):
    """
    Each (r,c) is a single node in the graph.
    We'll use networkx to create a 2D layout of visited nodes,
    so the resulting graph can expand in 2D.
    """
    env = RoomFloorEnv(layout)
    tdb = MockTDB(layout, K=10, seed=42)

    frames = []
    obs = env.reset()  # initial cell

    # Initialize graph with an initial node for obs
    # We'll store edges as we discover them
    G = nx.Graph()
    G.add_node(obs)  # the node is simply (r,c)

    # We'll keep a dict for TDB codes: cell2code[(r,c)] -> int
    cell2code = {}
    cell2code[obs] = tdb.get_code(*obs)

    for step_i in range(num_steps):
        # 1) Move agent
        action = random.choice([0,1,2,3])
        new_obs = env.step(action)

        # If new_obs isn't in G, add it
        if not G.has_node(new_obs):
            G.add_node(new_obs)
            cell2code[new_obs] = tdb.get_code(*new_obs)

        # Add an edge from old cell => new cell
        G.add_edge(obs, new_obs)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Left: floor plan
        draw_floor_plan(axes[0], layout, new_obs)
        code_new = cell2code[new_obs]
        axes[0].set_title(f"Step {step_i+1}: Agent at {new_obs}, code={code_new}")

        # Right: networkx 2D layout
        axes[1].set_title("Visited Cells (spring layout)")
        pos = nx.spring_layout(G, k=0.5, iterations=30)
        # if you want continuity in positions, you can keep 'pos' in a variable outside the loop and update it

        # We color the new node red, others blue
        node_colors = []
        for node in G.nodes():
            if node == new_obs:
                node_colors.append('red')
            else:
                node_colors.append('blue')

        # We'll label each node as (r,c): code
        labels = {}
        for node in G.nodes():
            r, c = node
            lbl_code = cell2code.get(node, -1)
            labels[node] = f"({r},{c}):C{lbl_code}"

        nx.draw(G, pos, ax=axes[1],
                node_color=node_colors,
                with_labels=True, labels=labels,
                font_size=8)

        axes[1].axis('off')
        fig.tight_layout()

        # Convert to RGBA array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        # If you have a high-DPI mismatch, you can do w *= 2; h *= 2
        w *= 2
        h *= 2
        raw_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = raw_data.reshape((h, w, 4))
        frames.append(frame)

        plt.close(fig)
        obs = new_obs

    return frames


def main():
    # Layout with multiple rooms (0=free, 1=wall)
    layout = [
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    ]

    frames = run_simulation(layout, num_steps=30)
    imageio.mimsave('room_floor_exploration.gif', frames, fps=0.5)
    print("Saved room_floor_exploration.gif.")

if __name__ == '__main__':
    # You could fix DPI to avoid mismatch:
    # plt.rcParams['figure.dpi'] = 96
    main()
