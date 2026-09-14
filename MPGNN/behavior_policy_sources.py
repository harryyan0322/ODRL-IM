"""Public upstream repositories used to generate logged behavior data.

The upstream implementations are not vendored: their licenses, model files,
and build requirements remain with their original authors.  This project
accepts local paths explicitly and records the corresponding repository in
the generated offline-data metadata.
"""

BEHAVIOR_POLICY_SOURCES = {
    "S2V_DQN": {
        "method": "S2V-DQN",
        "repository": "https://github.com/Dtrycode/ToupleGDD.git",
    },
    "Tripling": {
        "method": "ToupleGDD",
        "repository": "https://github.com/Dtrycode/ToupleGDD.git",
    },
    "PIANO": {
        "method": "PIANO",
        "repository": "https://github.com/lihuixidian/PIANO.git",
    },
}
