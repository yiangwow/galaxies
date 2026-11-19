# CLAUDE.md: galaxies

Galaxy collisions with real gravity, live on the GPU. One C++ file, OpenGL 4.6 compute shaders.
Started 2026-11-19. Repo: `yiangwow/galaxies` (private), remote `origin` over SSH host `github-alt`. Feature branch + PR, never push to main.
The creative direction is open: the owner decides it. Do not add game rules on your own.

## Build and run

`make`, then `./galaxies` (full screen; `--window` for a window). Needs glfw, libepoxy, g++.

Click = throw a galaxy in. Mouse = look. Wheel = zoom. `1-9 0 - = Backspace` = family of the
next galaxy. `R` start over, `F` full screen, `Space` pause, `Esc` quit.

## How it works (all in `galaxies.cpp`)

| Part | What |
|---|---|
| Gravity | Particle-mesh. Stars deposit mass on a 256^3 grid (`CS_UPDATE`), a multigrid V-cycle solves the Poisson equation (`CS_SMOOTH`, `CS_RESTRICT`, `CS_PROLONG`), `CS_GRAD` writes the force texture, each star reads its pull from it. One V-cycle per step, warm-started. |
| Particles | 4 slots x (1,048,576 stars + 262,144 dark-matter particles). A 5th galaxy replaces the oldest slot. |
| Galaxies | 13 families (`F_*`). `roll()` picks random parameters and colours, `buildGalaxy()` makes the particles on the CPU. Two worker threads roll the next one of each family in advance. |
| Drawing | Additive points into a float picture, bloom, asinh stretch (`FS_FINAL`). Supernova flashes and "young blue stars where it is crowded" are in `VS_STARS`, visual only. |
| Centring | `spawn()` kicks all stars so total momentum stays zero and slides the centre of mass back to the middle. |

## Checking (no window needed)

- `./galaxies --test` : one point mass, the solved pull must give ratio ~1.00 from r=16 up.
- `./galaxies --shot out.ppm --types 1,9 --seed 11 --steps 1100 --throw 150 --w 1920 --h 1080`
  then `ffmpeg -i out.ppm out.png`. `--types` = family numbers 1-13 (first galaxy, then throws).
- `--frames DIR --every N` dumps a frame every N steps. `--bench` prints ms per step.
- `GALAXIES_FPS=1 ./galaxies` prints the frame rate.

Measured 2026-11-19 on the RX 9070 XT: 3.5 ms per step with 5.2 M particles, about 150 fps at
2560x1440, about 11% of one CPU core, 181 MB RAM.

## Known limits

- Rings and bars are seeded shapes, not exact equilibria: they drift after a few turns.
- The box has zero-potential walls; stars that leave it fly free.
- Supernovae: checked by me on a recorded clip (`clips/merger.mp4`), the owner has not judged them yet. They fire in discs, arms and rings, not in the cores.
