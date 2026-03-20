# map.py
import numpy as np
import matplotlib.pyplot as plt

ROAD = 0
BUILDING = 1
GRASS = 2


def generate_city_map(
    size=100,
    block=10,
    road_w=2,
    n_grass=60,
    grass_min=3,
    grass_max=9,
    extra_roads=6,
    seed=None,
    show=False,
    show_nodes=True,
):
    rng = np.random.default_rng(seed)

    # 1) Start: everything is building
    grid = np.full((size, size), BUILDING, dtype=np.int8)

    # Centerlines for road bands
    x_centers = []  # vertical roads (columns) -> x
    y_centers = []  # horizontal roads (rows)  -> y

    # 2) Main roads
    for r in range(0, size, block):
        r0 = r
        r1 = min(size, r + road_w)
        grid[r0:r1, :] = ROAD
        y_centers.append(r0 + (road_w - 1) / 2.0)

    for c in range(0, size, block):
        c0 = c
        c1 = min(size, c + road_w)
        grid[:, c0:c1] = ROAD
        x_centers.append(c0 + (road_w - 1) / 2.0)

    # 2b) Extra random roads (choose starts so full width fits)
    for _ in range(extra_roads):
        if rng.random() < 0.5:
            r0 = int(rng.integers(0, size - road_w + 1))
            grid[r0:r0 + road_w, :] = ROAD
            y_centers.append(r0 + (road_w - 1) / 2.0)
        else:
            c0 = int(rng.integers(0, size - road_w + 1))
            grid[:, c0:c0 + road_w] = ROAD
            x_centers.append(c0 + (road_w - 1) / 2.0)

    # De-duplicate centerlines
    x_centers = sorted(set(x_centers))
    y_centers = sorted(set(y_centers))

    # Build intersection nodes as paired lists (x_nodes[k], y_nodes[k])
    x_nodes = []
    y_nodes = []
    for y in y_centers:
        for x in x_centers:
            x_nodes.append(x)
            y_nodes.append(y)

    # 3) Grass rectangles (never overwrite roads)
    for _ in range(n_grass):
        h = int(rng.integers(grass_min, grass_max))
        w = int(rng.integers(grass_min, grass_max))
        r0 = int(rng.integers(0, size - h))
        c0 = int(rng.integers(0, size - w))

        patch = grid[r0:r0 + h, c0:c0 + w]
        patch[patch != ROAD] = GRASS  # modifies grid via view

    # Optional visualization
    if show:
        plt.figure()
        plt.imshow(grid, origin="lower")
        if show_nodes:
            plt.scatter(x_nodes, y_nodes, marker=".", s=15)
        plt.title("City Map + Intersection Nodes")
        plt.show()

    return grid, x_nodes, y_nodes
