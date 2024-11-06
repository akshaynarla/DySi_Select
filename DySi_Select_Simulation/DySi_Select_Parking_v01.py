#!/usr/bin/env python3
#
# ==============================================================================
#
# Copyright 2023 IAS, Uni Stuttgart
#
# DySi_Select_Parking_v01 python module is used for prototyping the
# DySi_Select concept. This module provides the spawns parked vehicles in 
# parking spots or on the kerbs of Town01,02 and 03.
# ==============================================================================
"""DySi_Select_Parking_v01 python module is used for prototyping the 
DySi_Select concept. This module provides the spawns parked vehicles in 
parking spots or on the kerbs of Town01,02 and 03.
"""

import carla
import random


def park_vehicles():
    """Interface to spawn parked cars in Towns 01,02 and 03
    """
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode == True
        
        parking = []
        vehicle_bp = []
        town = world.get_map().name
        print(town)
        # get random spawn points on the town
        spawn_points = world.get_map().get_spawn_points()
        blueprint_library = world.get_blueprint_library()

        # random spawning of vehicles at random points 
        models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']
        for vehicle in blueprint_library.filter('*vehicle*'):
            if any(model in vehicle.id for model in models):
                vehicle_bp.append(vehicle)
        
        if town == 'Carla/Maps/Town03':
            spawn_points[1].location += carla.Location(y=4)
            spawn_points[42].location += carla.Location(x=-4) 
            spawn_points[57].location += carla.Location(x=4)
            spawn_points[65].location += carla.Location(x=-4)
            spawn_points[49].location += carla.Location(y=4)
            spawn_points[94].location += carla.Location(y=4)
        elif town == 'Carla/Maps/Town02':
            spawn_points[6].location += carla.Location(x=4)
            spawn_points[11].location += carla.Location(x=-4) 
            spawn_points[37].location += carla.Location(y=-4)
            spawn_points[62].location += carla.Location(y=-4)
            spawn_points[78].location += carla.Location(x=-4)
            spawn_points[69].location += carla.Location(y=-4)
        elif town == 'Carla/Maps/Town01':
            spawn_points[0].location += carla.Location(x=6)
            spawn_points[2].location += carla.Location(y=-4) 
            spawn_points[41].location += carla.Location(y=-6)
            spawn_points[55].location += carla.Location(x=4)
            spawn_points[75].location += carla.Location(y=-4)
            spawn_points[88].location += carla.Location(y=4)
        for i in range(0,5):
            if town == 'Carla/Maps/Town03':
                spawn_points[1].location += carla.Location(x=8)
                npc = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[1])
                spawn_points[42].location += carla.Location(y=4)
                npc2 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[42])
                spawn_points[57].location += carla.Location(y=-4)
                npc3 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[57])
                spawn_points[65].location += carla.Location(y=6)
                npc4 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[65])
                spawn_points[49].location += carla.Location(x=5)
                npc5 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[49])
                spawn_points[94].location += carla.Location(x=7)
                npc6 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[94])
            elif town == 'Carla/Maps/Town02':
                spawn_points[6].location += carla.Location(y=6)
                npc = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[6])
                spawn_points[11].location += carla.Location(y=4)
                npc2 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[11])
                spawn_points[37].location += carla.Location(x=6)
                npc3 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[37])
                spawn_points[62].location += carla.Location(x=-6)
                npc4 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[62])
                spawn_points[78].location += carla.Location(y=-4)
                npc5 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[78])
                spawn_points[69].location += carla.Location(x=-6)
                npc6 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[69])
            elif town == 'Carla/Maps/Town01':
                spawn_points[0].location += carla.Location(y=6)
                npc = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[0])
                spawn_points[2].location += carla.Location(x=-6)
                npc2 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[2])
                spawn_points[41].location += carla.Location(x=6)
                npc3 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[41])
                spawn_points[55].location += carla.Location(y=-6)
                npc4 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[55])
                spawn_points[75].location += carla.Location(x=-6)
                npc5 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[75])
                spawn_points[88].location += carla.Location(x=6)
                npc6 = world.try_spawn_actor(random.choice(vehicle_bp), transform=spawn_points[88])                
            # only append if the spawned actor is not None
            # else causes problems for destroying the actor
            if npc is not None:
                parking.append(npc)
            if npc2 is not None:    
                parking.append(npc2)
            if npc3 is not None:
                parking.append(npc3)
            if npc4 is not None:
                parking.append(npc4)
            if npc5 is not None:
                parking.append(npc5) 
            if npc6 is not None:                
                parking.append(npc6)
        p_list = parking
        print('spawned %d vehicles, press Ctrl+C to exit.' % (len(p_list)))
        while True:
            if settings.synchronous_mode == True:
                world.tick()
            else:
                world.wait_for_tick()     
    finally:
        # cleanup and destroy the actors
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print('\ndestroying %d vehicles' % len(p_list))
        # client.apply_batch_sync([carla.command.DestroyActor(x) for x in p_list])
        for x in p_list:
            x.destroy()
        
        
# ==============================================================================
# Start of simulation
# ==============================================================================
if __name__ == '__main__':
    try:
        park_vehicles()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')