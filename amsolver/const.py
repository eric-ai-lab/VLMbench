#Copy From the rlbench: https://github.com/stepjam/RLBench
#seen
seen_colors = [
    ('red', (1.0, 0.0, 0.0)),
    ('maroon', (0.5, 0.0, 0.0)),
    ('lime', (0.0, 1.0, 0.0)),
    ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)),
    ('orange', (1.0, 0.5, 0.0)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
]

all_colors = [
    ('red', (1.0, 0.0, 0.0)),
    ('maroon', (0.5, 0.0, 0.0)),
    ('lime', (0.0, 1.0, 0.0)),
    ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)),
    ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
]
#unseen
unseen_colors = [
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
]
#seen objects
seen_object_shapes = {
    "star":{
        "path":"star/star_normal/star_normal.ttm"
    },
    "triangular":{
        "path":"triangular/triangular_normal/triangular_normal.ttm"
    },
    "cylinder":{
        "path":"cylinder/cylinder_normal/cylinder_normal.ttm"
    },
    "cube":{
        "path":"cube/cube_basic/cube_basic.ttm"
    },
    "letter_r":{
        "path":"letters/letter_r/letter_r.ttm"
    },
    "letter_a":{
        "path":"letters/letter_a/letter_a.ttm"
    },
    "letter_v":{
        "path":"letters/letter_v/letter_v.ttm"
    },
    "pentagon":{
        "path":"basic_shapes/pentagon/pentagon.ttm"
    }
}
#unseen objects
unseen_object_shapes = {
    "moon":{
        "path":"moon/moon_normal/moon_normal.ttm"
    },
    "letter_t":{
        "path":"letters/letter_t/letter_t.ttm"
    },
    "cross":{
        "path":"basic_shapes/cross/cross.ttm"
    },
    "flower":{
        "path":"basic_shapes/flower/flower.ttm"
    }
}

seen_planes ={
    "rectangle":{
        "path":"basic_shapes/plane_rectangle.ttm",
        "directional": True
    },
    "round":{
        "path":"basic_shapes/plane_round.ttm",
        "directional": False
    },
    "star":{
        "path":"basic_shapes/plane_star.ttm",
        "directional": False
    },
    "triangle":{
        "path":"basic_shapes/plane_triangle.ttm",
        "directional": True
    }
}

seen_sorter_objects = {
    "star":{
        "path":"star/star_normal/star_normal.ttm"
    },
    "triangular":{
        "path":"triangular/triangular_normal/triangular_normal.ttm"
    },
    "cylinder":{
        "path":"cylinder/cylinder_normal/cylinder_normal.ttm"
    },
    "cube":{
        "path":"cube/cube_basic/cube_basic.ttm"
    }
}

unseen_sorter_objects = {
    "moon":{
        "path":"moon/moon_normal/moon_normal.ttm"
    },
}

colors = seen_colors
object_shapes = seen_object_shapes
planes = seen_planes
sorter_objects = seen_sorter_objects