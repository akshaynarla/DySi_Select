#!/usr/bin/env python3
# 
# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# DySi_Sel_v01 python module is used for prototyping the client for
# DySi_Select concept. This module sets up the simulation of autonomous vehicles 
# and the surrounding environment, to be used as input for Cam2BEV. This will 
# further be used for demonstrating the identified situation by the ego-vehicle.
# This module is a modified version of PythonAPI/examples/manual_control.py from 
# the CARLA repository.
# ==============================================================================
"""
Welcome to CARLA DySi_Select

    CTRL + W     : toggle constant velocity mode at 50 km/h
    N            : next sensor
    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""
from __future__ import print_function
import glob
import os
import sys

import logging
import random
import weakref
import numpy as np
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from DySi_Sel_utils import *

from Cam2BEV.predict_bev import predict_bev, get_bev_network
from SIN.src.predict_situation import predict_situation, get_network_sin
from SIN.src.utils.data_utils import load_image
from SemSeg.sem_seg import predict_semseg, get_network_semseg
from evaluate_real_dataset import data_select

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SLASH
    from pygame.locals import K_h
    from pygame.locals import K_n
    from pygame.locals import K_q
    from pygame.locals import K_w
    from pygame.locals import K_F1
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# ==============================================================================
# World class provides an instance of the world to which the client vehicle is
# to be spawned. All necessary conditions for the ego-vehicle/client is set here.
#
# Returns: an instance of the client in the CARLA world
# ==============================================================================
class World(object):
    """World class provides an instance of the world to which the client vehicle is
    to be spawned. All necessary conditions for the ego-vehicle/client is set here.

    """
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.camera_manager = None
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.constant_velocity_enabled = False

    # spawn the client in the world
    def restart(self):        
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        # Get the blueprint of "Mercedes-Benz Coupe", a common urban scenario sedan
        blueprint = self.world.get_blueprint_library().find('vehicle.mercedes.coupe')
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        
        # Set up the sensors.
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def modify_vehicle_physics(self, vehicle):
        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)

    def tick(self, clock):
        self.hud.tick(self, clock)

    # render the carla world on pygame window
    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    # destroy and cleanup CARLA world
    def destroy(self):
        sensors = [
            self.camera_manager.sensor
            ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# SimControl provides an instance of necessary control of simulation.
# The instance of this class allows the client to switch between sensors and
# change certain simulation parameters.
#
# Returns: an instance of simulation control
# ==============================================================================
class SimControl(object):
    """SimControl provides an instance of necessary control of simulation.
    The instance of this class allows the client to switch between sensors and change certain simulation parameters.

    """
    # initialize the simulation in autopilot (to validate in a autonomous vehicle environment)
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    # interface for reading user inputs from keyboard
    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    # to-do: ensure it follows rules, if not remove
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                        world.player.set_autopilot(self._autopilot_enabled)
                    else:
                        world.player.set_target_velocity(carla.Vector3D(13, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 50 km/h")

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


class CameraManager(object):
    """CameraManager provides an instance of necessary camera sensors for use 
    in the simulation. By default, a Semantic Segmentation camera in CItyscapes palette 
    will be run while initializing the simulation. 
    Alternatively, a RGB Camera is provided which can be visualized by pressing "n" 
    after start of simulation.
    This is provided in the main python file instead of utils to avoid use in other modules.

    """
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = True
        self.transform_index = 1
        # more cameras can be defined here
        self.sensors = [
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}]
            ]
        # initialize defined sensor/s
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            item.append(bp)
        self.index = None

    # spawn the defined sensors
    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            # the camera transform x,y,z here is from Cam2BEV. 
            # This can be changed based on need. 
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                (carla.Transform(carla.Location(x=1.84452,y=-0.096, z=1.1273), carla.Rotation(0,0,0))),
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            frame_proc = 0
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    # used to switch sensor (pressing "n")
    def next_sensor(self):
        self.set_sensor(self.index + 1)

    # render the camera output on pygame window
    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    # process the output from sensor and save to disk, 
    # to be used for evaluation and further processing
    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording and ((image.frame % 10) == 0):
            if(self.sensors[self.index][2] == 'Camera RGB'):
                # takes too much resource in CPU, slows down the process
                img_file = f"rgb_{self._parent.id}_{image.frame}"
                img_path = os.path.join("_out/rgb",img_file)
                img_loc = image.save_to_disk(img_path)
                # get semantic segmentation image
                frame2, sem_loc = predict_semseg(model=semseg_nw, original_img=img_loc)
            else:
                sem_file = f"Sem_{self._parent.id}_{image.frame}"
                sem_path = os.path.join("_out/sem",sem_file)
                sem_loc = image.save_to_disk(sem_path)
            # get bev image
            frame3, bev_loc = predict_bev(model=bev_nw, one_hotd_label=label,
                                          sem_img=sem_loc, backbone=backbone_m)
            # get SIN
            bev = load_image(bev_loc)
            # display the identified situation on window
            situation = predict_situation(model=sin_nw, bev_img=bev)
            data_tx = data_select(situation)
            cv2.putText(frame3, situation, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)
            cv2.putText(frame3, data_tx, (10,45), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)
            # Create the filename
            file_name = f"SIN_{self._parent.id}_{image.frame}.png"
            # Create the full file path within the new run folder
            file_path = os.path.join("_out/test", file_name)
            cv2.imwrite(file_path, cv2.cvtColor(frame3,cv2.COLOR_RGB2BGR))
            

def game_loop(args):
    """game_loop() interface initializes the ego-vehicle client 
    and runs the necessary instances for simulation. 

    Args:
        args (config): parsed configuration parameters from CLI
    """
    # initialize the pygame window for visualizing the simulation
    pygame.init()
    pygame.font.init()
    world = None
    synchronous_master = False

    try:
        # connect the client to running CARLA server
        client = carla.Client(args.host, args.port)
        client.set_timeout(25.0)

        sim_world = client.get_world()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_random_device_seed(42)
            
        # set the parameters of the pygame display window
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        # Initialize the HUD for displaying details about the 
        # surroundings of ego-vehicle
        hud = HUD(args.width, args.height)
        
        # Initialize and set the parameters for the client in CARLA world. 
        # Necessary sensors, transforms, spawn points are set here.
        world = World(sim_world, hud, args)
        
        controller = SimControl(world, args.autopilot)
        
        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()
        
        # Run the client continuosly until interrupted or quit
        clock = pygame.time.Clock()
        
        while True:
            if args.sync:
                sim_world.tick()
            else:
                sim_world.wait_for_tick()
            clock.tick_busy_loop(args.fps)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            # display the identified situation on window
            pygame.display.flip()

    finally:
        if args.sync and synchronous_master:
            settings = sim_world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            sim_world.apply_settings(settings)
        # destroy all actors and cleanup the CARLA world
        if world is not None:
            world.destroy()
        pygame.quit()


def main():
    """main interface of the DySi_Select simulation.
    """
    # get arguments from CLI
    args = get_args()
    # resolution of the output window
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    # log the server details
    logging.info('listening to server %s:%s', args.host, args.port)
    # print necessary information on CLI for manipulating simulation
    print(__doc__)

    global semseg_nw, bev_nw, sin_nw, label, backbone_m
    backbone_m = args.backbone
    semseg_nw = get_network_semseg()
    bev_nw, label = get_bev_network(backbone=backbone_m,weights_dir=args.weight1)
    sin_nw = get_network_sin(weights_dir=args.weight2)
    
    try:
        # continuously run simulation loop
        game_loop(args)
    except KeyboardInterrupt:
        # Stops simulation in case of interruption with Ctrl+C
        print('\nCancelled by user. Bye!')

# ==============================================================================
# Start of simulation
# ==============================================================================
if __name__ == '__main__':
    main()
