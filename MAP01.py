from vizdoom import DoomGame, Mode, Button, GameVariable, ScreenResolution, ScreenFormat
import random
import time
import pygame

def setup_game():
    game = DoomGame()
    game.set_doom_scenario_path("MAP01.wad")
    game.set_doom_map("map01")  # Adjust if your map name is different

    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)

    game.add_available_button(Button.MOVE_FORWARD)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)

    game.set_episode_timeout(2000)
    game.set_episode_start_time(10)
    game.set_window_visible(True)  # Set to True if you want to see the game when running the container with a display

    game.init()
    return game

def navigate_maze(game):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer
        
        # Simple random movement
        action = random.choice([0, 1, 2])  # 0: forward, 1: left, 2: right
        reward = game.make_action([action == 0, action == 1, action == 2])
        
        if game.is_player_dead():
            break

        time.sleep(0.028)  # ~35 fps

    print("Episode finished. Total reward:", game.get_total_reward())

if __name__ == "__main__":
    game = setup_game()
    navigate_maze(game)
    game.close()
