#!/usr/bin/env python3
# 
# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# DySi_Sel_Traffic_v01 python module is used for spawning the necessary traffic 
# for evaluation of DySi_Select concept. Due to the use of "seed", the traffic is
# always spawned at the same points i.e. repeatability ensured.
# ==============================================================================
""" Spawning Traffic into the simulation

    DySi_Sel_Traffic python module is used for spawning the necessary traffic 
    for evaluation of DySi_Select concept. Due to the use of "seed", the traffic is 
    always spawned at the same points i.e. repeatability ensured.
"""
import glob
import os
import sys
import time
import argparse
import logging
from numpy import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import VehicleLightState as vls


def spawn_traffic():
    """spawn_traffic interface of the DySi_Sel_Traffic. This is used to spawn traffic onto 
    the CARLA world. The arguments to be parsed via CLI are available within the 
    module itself to ensure non-interference between different involved modules.
    """
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=100,
        type=int,
        help='number of vehicles (default: 100)')
    argparser.add_argument(
        '--safe',
        metavar='N',
        default=True,
        type=bool,
        help='avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enable')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        default = 42,
        type=int,
        help='Random device seed')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable car lights')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    # connect the client python module to CARLA server
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()
        # traffic manager manages the traffic and ensures each vehicle is 
        # spawned at same point using the seed
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)
        # set synchronous mode
        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False
        # get the blueprints for the vehicle
        vehicle_bp = []
        blueprints = world.get_blueprint_library().filter(args.filterv)
        models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']
        for vehicle in blueprints:
            if any(model in vehicle.id for model in models):
                vehicle_bp.append(vehicle)

        blueprints = sorted(vehicle_bp, key=lambda bp: bp.id)
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # spawn vehicles onto CARLA world
        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # Spawn necessary vehicle traffic
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            if args.car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        print('spawned %d vehicles, press Ctrl+C to exit.' % (len(vehicles_list)))

        # Use of parameters for traffic manager
        traffic_manager.global_percentage_speed_difference(30.0)

        while True:
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    finally:
        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        # cleanup and destroy the actors
        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

# ==============================================================================
# Start of simulation
# ==============================================================================
if __name__ == '__main__':
    try:
        spawn_traffic()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
