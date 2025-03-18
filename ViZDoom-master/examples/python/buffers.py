#!/usr/bin/env python3

#####################################################################
# This script presents different buffers and formats.
# OpenCV is used here to display images, install it or remove any
# references to cv2
# Configuration is loaded from "../../scenarios/basic.cfg" file.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from argparse import ArgumentParser
import os
from random import choice
import vizdoom as vzd

#DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
DEFAULT_CONFIG = "../../scenarios/basic.cfg"

import cv2

if __name__ == "__main__":

    parser = ArgumentParser("ViZDoom example showing different buffers (screen, depth, labels).")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")

    args = parser.parse_args()

    game = vzd.DoomGame()

    # Use other config file if you wish.
    game.load_config(args.config)

    game.set_console_enabled(True)
    game.set_window_visible(False)

    # Just uncomment desired format for screen buffer (and map buffer).
    # The last uncommented will be applied.
    # Formats with C (CRCGCB and CBCGCR) were omitted cause they are not cv2 friendly.
    # Default format is ScreenFormat.CRCGCB.

    # OpenCV uses a BGR colorspace by default.
    game.set_screen_format(vzd.ScreenFormat.BGR24)

    # game.set_screen_format(ScreenFormat.RGB24)
    # game.set_screen_format(ScreenFormat.RGBA32)
    # game.set_screen_format(ScreenFormat.ARGB32)
    # game.set_screen_format(ScreenFormat.BGRA32)
    # game.set_screen_format(ScreenFormat.ABGR32)
    # game.set_screen_format(ScreenFormat.GRAY8)

    # Raw Doom buffer with palette's values. This one makes no sense in particular
    # game.set_screen_format(ScreenFormat.DOOM_256_COLORS)

    # Sets resolution for all buffers.
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of he current episode/level .
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    game.set_render_hud(True)
    game.set_render_minimal_hud(False)

    game.set_mode(vzd.Mode.SPECTATOR)
    game.init()

    actions = [[True, False, False], [False, True, False], [False, False, True]]
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    game.set_render_hud(False)
    game.set_render_minimal_hud(False)(r(i + 1))

    # Entire map is shown        # Not needed for the first episode but the loop is nicer.
    game.new_episode()
    while not game.is_episode_finished():
            # Gets the state and possibly do something with it
        state = game.get_state()

    ##############################################################################################################################################################
        cv2.imshow('ViZDoom Screen Buffer', screen)

            # Depth buffer, always in 8-bit gray channel format.
            # This is most fun. It looks best if you inverse colors.
        depth = state.depth_buffer
        if depth is not None:
            cv2.imshow('ViZDoom Depth Buffer', depth)

            # Labels buffer, always in 8-bit gray channel format.
            # Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
            # Labels data are available in state.labels, also see labels.py example.
        labels = state.labels_buffer
        if labels is not None:
            cv2.imshow('ViZDoom Labels Buffer', labels)

            # Map buffer, in the same format as screen buffer.
            # Shows top down map of the current episode/level.
        automap = state.automap_buffer
        if automap is not None:
            cv2.imshow('ViZDoom Map Buffer', automap)

        cv2.waitKey(int(sleep_time * 1000))

        game.make_action(choice(actions))

        print("State #" + str(state.number))
        print(state.game_variables)
        print("=====================")

    print("Episode finished!")
    print("************************")

    cv2.destroyAllWindows()
print(f"Using calibrated map center: ({map_center_x}, {map_center_y}) and scale factor: {scale_factor}")        # Initialize controller for trajectory following with improved parameters    controller = ImprovedPurePursuitController(lookahead_distance=50)    if path:        controller.set_path(path)            # Set the calibrated angle offset    controller.angle_offset = angle_offset    episodes = 1    sleep_time = 0.028    # Main loop with calibrated values    for i in range(episodes):        print("Episode #" + str(i + 1))        game.new_episode()        while not game.is_episode_finished():            # Get current state            state = game.get_state()            if state is None:                continue            # Get current position and angle            doom_x = state.game_variables[0]            doom_y = state.game_variables[1]            angle = state.game_variables[2]                        # Convert Doom coordinates to pixel coordinates            current_x, current_y = doom_to_pixel(doom_x, doom_y, map_center_x, map_center_y, scale_factor)                        # Get action from controller            action = [False, False, True, False, 0]  # Default: move forward            if path:                action = controller.update(current_x, current_y, angle)                        # Visualize current position on the path            if path:                live_vis = path_vis.copy()                cv2.circle(live_vis, (current_x, current_y), 3, (0, 255, 255), -1)                                # Draw line to target waypoint                if controller.target_idx < len(path):                    target = path[controller.target_idx]                    cv2.line(live_vis, (current_x, current_y),                             (int(target[0]), int(target[1])), (0, 0, 255), 1)                                cv2.imshow('Current Position', live_vis)                        # Show the automap            automap = state.automap_buffer            if automap is not None:                cv2.imshow('ViZDoom Map Buffer', automap)                        # Execute the action            game.make_action(action)                        # Debug info            print(f"Action: {action}")            print("=====================")                        cv2.waitKey(int(sleep_time * 1000))        print("Episode finished!")        print("************************")    cv2.destroyAllWindows()