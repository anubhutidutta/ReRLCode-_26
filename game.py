import pygame
from env import StudentEnv
from agent import Agent

pygame.init()

# Screen
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Student RL Game")

clock = pygame.time.Clock()
#colors
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

# -------- ANIMATION LOADING --------
def load_animation(path):
    sheet = pygame.image.load(path).convert_alpha()
    frame_width = 128
    frame_height = 128

    sheet_width = sheet.get_width()
    frame_count = sheet_width // frame_width

    frames = []
    for i in range(frame_count):
        frame = sheet.subsurface((i * frame_width, 0, frame_width, frame_height))
        frame = pygame.transform.scale(frame, (80, 80))
        frames.append(frame)

    return frames

idle_frames = load_animation("Idle.png")
run_frames = load_animation("Run.png")
walk_frames = load_animation("Walk.png")
jump_frames = load_animation("Jump.png")
hurt_frames = load_animation("Hurt.png")
dead_frames = load_animation("Dead.png")

frame_index = 0

# RL setup
env = StudentEnv()
agent = Agent(4, 5)
agent.load()

state = env.reset()

# Player
player_x = 100
player_y = 260
speed = 0
world_x = 0
# Timing for RL
last_update = 0
action = 0
done = False
# -------- BACKGROUND LAYERS --------
bg_far = pygame.Surface((WIDTH, HEIGHT))
bg_far.fill((180, 220, 255))  # sky

bg_mid = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
bg_mid.fill((0, 0, 0, 0))
for i in range(5):
    pygame.draw.circle(bg_mid, (150, 200, 255), (i * 200, 150), 80)

bg_near = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
bg_near.fill((0, 0, 0, 0))
for i in range(8):
    pygame.draw.rect(bg_near, (100, 180, 100), (i * 120, 250, 80, 150))
running = True
while running:
    clock.tick(60)
    bg_mid_x = (world_x * 0.3) % WIDTH
    bg_near_x = (world_x * 0.6) % WIDTH
    # FAR (slowest)
    screen.blit(bg_far, (0, 0))

    # MID
    screen.blit(bg_mid, (-bg_mid_x, 0))
    screen.blit(bg_mid, (-bg_mid_x + WIDTH, 0))

    # NEAR
    screen.blit(bg_near, (-bg_near_x, 0))
    screen.blit(bg_near, (-bg_near_x + WIDTH, 0))
    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # -------- RL UPDATE (slow and stable) --------
    current_time = pygame.time.get_ticks()
    if current_time - last_update > 80:
        action = agent.act(state)
        state, reward, done = env.step(action)
        last_update = current_time

    # -------- MOVEMENT --------
    if action == 0:      # run
        target_speed = 3
    elif action == 1:    # walk
        target_speed = 1.5
    else:                # rest
        target_speed = 0

    # limit + smooth
    speed = min(speed, 3)
    speed *= 0.90
    #move world instead of player 
    world_x += speed
    # trigger game over


    if player_x < 0:
        player_x = 0

    # -------- UI BARS --------
    fuel = state[0] * 100
    pygame.draw.rect(screen, GREEN, (50, 50, fuel * 2, 20))

    attendance = state[1] * 100
    pygame.draw.rect(screen, RED, (50, 80, attendance * 2, 20))

    # -------- ANIMATION --------
    if done:
        current_anim = dead_frames
    elif state[0] < 0.3:
        current_anim = hurt_frames
    elif action == 0:
        current_anim = run_frames
    elif action == 1:
        current_anim = walk_frames
    else:
        current_anim = idle_frames

    frame_index += 0.08 + abs(speed) * 0.03
    if frame_index >= len(current_anim):
        frame_index = 0

    current_frame = current_anim[int(frame_index)]
    screen.blit(current_frame, (player_x, player_y))

    # Ground
    for i in range(0, WIDTH, 40):
        pygame.draw.rect(screen, (200, 200, 200),
                     (i - (world_x % 40), 340, 40, 20))

    pygame.display.update()

    if done:
        font = pygame.font.Font(None, 60)
        text = font.render("Game Over", True, (0, 0, 0))
        screen.blit(text, (300, 150))
        pygame.display.update()
        pygame.time.delay(1500)
        running = False
    

pygame.quit()