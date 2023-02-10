from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.data import Difficulty, Race  # difficulty for bots, race for the 1 of 3 races
from sc2.main import run_game  # function that facilitates actually running the agents in games
from sc2.player import Bot, Computer  #wrapper for whether or not the agent is one of your bots, or a "computer" player
from sc2 import maps  # maps method for loading maps to play in.
from sc2.ids.unit_typeid import UnitTypeId
import random
import cv2
import math
import numpy as np
import sys
import pickle
import time
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY, SCV, DRONE, ROBOTICSFACILITY, OBSERVER, \
 ZEALOT, STALKER

SAVE_REPLAY = True

total_steps = 10000 
steps_for_pun = np.linspace(0, 1, total_steps)
step_punishment = ((np.exp(steps_for_pun**3)/10) - 0.1)*10



class IncrediBot(BotAI): # inhereits from BotAI (part of BurnySC2)

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def build_scout(self):
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            print(len(self.units(OBSERVER)), self.cur_time/3)
            if self.can_afford(OBSERVER) and self.supply_left > 0:
                await self.do(rf.train(OBSERVER))
                break
        if len(self.units(ROBOTICSFACILITY)) == 0:
            pylon = self.units(PYLON).ready.noqueue.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                    await self.build(ROBOTICSFACILITY, near=pylon)


    async def build_worker(self):
        nexuses = self.units(NEXUS).ready.noqueue
        if nexuses.exists:
            if self.can_afford(PROBE):
                await self.do(random.choice(nexuses).train(PROBE))

    async def build_zealot(self):
        #if len(self.units(ZEALOT)) < (8 - self.cur_time): # how we can phase out zealots over time?
        gateways = self.units(GATEWAY).ready.noqueue
        if gateways.exists:
            if self.can_afford(ZEALOT):
                await self.do(random.choice(gateways).train(ZEALOT))

    async def build_gateway(self):
        #if len(self.units(GATEWAY)) < 5:
        pylon = self.units(PYLON).ready.noqueue.random
        if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
            await self.build(GATEWAY, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_voidray(self):
        stargates = self.units(STARGATE).ready.noqueue
        if stargates.exists:
            if self.can_afford(VOIDRAY):
                await self.do(random.choice(stargates).train(VOIDRAY))

    async def build_stalker(self):
        pylon = self.units(PYLON).ready.noqueue.random
        gateways = self.units(GATEWAY).ready
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if gateways.exists and cybernetics_cores.exists:
            if self.can_afford(STALKER):
                await self.do(random.choice(gateways).train(STALKER))

        if not cybernetics_cores.exists:
            if self.units(GATEWAY).ready.exists:
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_assimilator(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

    async def build_stargate(self):
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_pylon(self):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON) and not self.already_pending(PYLON):
                    await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))

    async def expand(self):
        try:
            if self.can_afford(NEXUS) and len(self.units(NEXUS)) < 3:
                await self.expand_now()
        except Exception as e:
            print(str(e))

    async def do_nothing(self):
        wait = random.randrange(7, 100)/100
        self.do_something_after = self.cur_time + wait


    async def on_step(self, iteration: int): # on_step is a method that is called every step of the game.
        no_action = True
        while no_action:
            try:
                with open('state_rwd_action.pkl', 'rb') as f:
                    state_rwd_action = pickle.load(f)

                    if state_rwd_action['action'] is None:
                        #print("No action yet")
                        no_action = True
                    else:
                        #print("Action found")
                        no_action = False
            except:
                pass


        await self.distribute_workers() # put idle workers back to work

        action = state_rwd_action['action']
        '''
        0: expand (ie: move to next spot, or build to 16 (minerals)+3 assemblers+3)
        1: build stargate (or up to one) (evenly)
        2: build voidray (evenly)
        3: send scout (evenly/random/closest to enemy?)
        4: attack (known buildings, units, then enemy base, just go in logical order.)
        5: voidray flee (back to base)
        '''

        # 0: expand (ie: move to next spot, or build to 16 (minerals)+3 assemblers+3)
        if action == 0:
            try:
                found_something = False
                if self.supply_left < 4:
                    # build pylons. 
                    if self.already_pending(UnitTypeId.PYLON) == 0:
                        if self.can_afford(UnitTypeId.PYLON):
                            # await self.build(UnitTypeId.PYLON, near=random.choice(self.townhalls))
                            await self.build_pylon()
                            found_something = True

                if not found_something:

                    for nexus in self.townhalls:
                        # get worker count for this nexus:
                        worker_count = len(self.workers.closer_than(10, nexus))
                        if worker_count < 22: # 16+3+3
                            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE):
                                # nexus.train(UnitTypeId.PROBE)
                                await self.build_worker()
                                found_something = True

                        # have we built enough assimilators?
                        # find vespene geysers
                        for geyser in self.state.vespene_geyser.closer_than(10, nexus):
                            # build assimilator if there isn't one already:
                            if not self.can_afford(UnitTypeId.ASSIMILATOR):
                                break
                            if not self.units.structure(UnitTypeId.ASSIMILATOR).closer_than(2.0, geyser).exists:
                                # await self.build(UnitTypeId.ASSIMILATOR, geyser)
                                await self.build_assimilator()
                                found_something = True

                if not found_something:
                    if self.already_pending(UnitTypeId.NEXUS) == 0 and self.can_afford(UnitTypeId.NEXUS):
                        # await self.expand_now()
                        await self.expand()


            except Exception as e:
                print(e)


        #1: build stargate (or up to one) (evenly)
        elif action == 1:
            try:
                # iterate thru all nexus and see if these buildings are close
                for nexus in self.townhalls:
                    # is there is not a gateway close:
                    if not self.units.structure(UnitTypeId.GATEWAY).closer_than(10, nexus).exists:
                        # if we can afford it:
                        if self.can_afford(UnitTypeId.GATEWAY) and self.already_pending(UnitTypeId.GATEWAY) == 0:
                            # build gateway
                            # await self.build(UnitTypeId.GATEWAY, near=nexus)
                            await self.build_gateway()
                        
                    # if the is not a cybernetics core close: // WRONG
                    # if not self.units.structure(UnitTypeId.CYBERNETICSCORE).closer_than(10, nexus).exists:
                    #     # if we can afford it:
                    #     if self.can_afford(UnitTypeId.CYBERNETICSCORE) and self.already_pending(UnitTypeId.CYBERNETICSCORE) == 0:
                    #         # build cybernetics core
                    #         await self.build(UnitTypeId.CYBERNETICSCORE, near=nexus)
                    # RIGHT - if cybernetics core doesn't exist
                    pylon = self.units(PYLON).ready.noqueue.random      
                    cybernetics_cores = self.units(CYBERNETICSCORE).ready
                    if not cybernetics_cores.exists:
                        if self.units(GATEWAY).ready.exists:
                            if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                                await self.build(CYBERNETICSCORE, near=pylon.position.towards(self.game_info.map_center, 5))

                    # # if there is not a stargate close:
                    # if not self.units.structure(UnitTypeId.STARGATE).closer_than(10, nexus).exists:
                    #     # if we can afford it:
                    #     if self.can_afford(UnitTypeId.STARGATE) and self.already_pending(UnitTypeId.STARGATE) == 0:
                    #         # build stargate
                    #         await self.build(UnitTypeId.STARGATE, near=nexus)

                    await self.build_stargate()

            except Exception as e:
                print(e)


        #2: build voidray (random stargate)
        elif action == 2:
            try:
                # if self.can_afford(UnitTypeId.VOIDRAY):
                #     for sg in self.units.structure(UnitTypeId.STARGATE).ready.idle:
                #         if self.can_afford(UnitTypeId.VOIDRAY):
                #             sg.train(UnitTypeId.VOIDRAY)
                await self.build_voidray()
            
            except Exception as e:
                print(e)

        #3: send scout
        elif action == 3:
            # are there any idle probes:
            try:
                self.last_sent
            except:
                self.last_sent = 0

            # if self.last_sent doesnt exist yet:
            if (iteration - self.last_sent) > 200:
                try:
                    if self.units(UnitTypeId.PROBE).idle.exists:
                        # pick one of these randomly:
                        probe = random.choice(self.units(UnitTypeId.PROBE).idle)
                    else:
                        probe = random.choice(self.units(UnitTypeId.PROBE))
                    # send probe towards enemy base:
                    probe.attack(self.enemy_start_locations[0])
                    self.last_sent = iteration

                except Exception as e:
                    pass


        #4: attack (known buildings, units, then enemy base, just go in logical order.)
        elif action == 4:
            try:
                # take all void rays and attack!
            
                for u in self.units(VOIDRAY).idle:
                    # if we can attack:
                    if self.known_enemy_units.closer_than(10, u):
                        # attack!
                        # u.attack(random.choice(self.known_enemy_units.closer_than(10, u)))
                        target=random.choice(self.known_enemy_units.closer_than(10, u))
                        await self.do(u.attack(target))

                    # if we can attack:
                    elif self.known_enemy_structures.closer_than(10, u):
                        # attack!
                        # u.attack(random.choice(self.known_enemy_structures.closer_than(10, u)))
                        target = random.choice(self.known_enemy_structures.closer_than(10, u))
                        await self.do(u.attack(target))
                    # any enemy units:
                    elif self.known_enemy_units:
                        # attack!
                        # u.attack(random.choice(self.known_enemy_units))
                        target = random.choice(self.known_enemy_units)
                        await self.do(u.attack(target))
                    # any enemy structures:
                    elif self.known_enemy_structures:
                        # attack!
                        # u.attack(random.choice(self.known_enemy_structures))
                        target = random.choice(self.known_enemy_structures)
                        await self.do(u.attack(target)) 

                    # if we can attack:
                    elif self.enemy_start_locations:
                        # attack!
                        # u.attack(self.enemy_start_locations[0])
                        target = self.enemy_start_locations[0]
                        await self.do(u.attack(target)) 

                    


                # for u in self.units(STALKER).idle:
                #     target = self.start_location
                #     # if we can attack:
                #     if self.known_enemy_units.closer_than(10, u):
                #         # attack!
                #         # u.attack(random.choice(self.known_enemy_units.closer_than(10, u)))
                #         target=random.choice(self.known_enemy_units.closer_than(10, u))

                #     # if we can attack:
                #     elif self.known_enemy_structures.closer_than(10, u):
                #         # attack!
                #         # u.attack(random.choice(self.known_enemy_structures.closer_than(10, u)))
                #         target = random.choice(self.known_enemy_structures.closer_than(10, u))
                #     # any enemy units:
                #     elif self.known_enemy_units:
                #         # attack!
                #         # u.attack(random.choice(self.known_enemy_units))
                #         target = random.choice(self.known_enemy_units)
                #     # any enemy structures:
                #     elif self.known_enemy_structures:
                #         # attack!
                #         # u.attack(random.choice(self.known_enemy_structures))
                #         target = random.choice(self.known_enemy_structures)
                #     # if we can attack:
                #     elif self.enemy_start_locations:
                #         # attack!
                #         # u.attack(self.enemy_start_locations[0])
                #         target = self.enemy_start_locations[0]

                #     await self.do(u.attack(target)) 

                # for u in self.units(ZEALOT).idle:
                #     target = self.start_location
                #     # if we can attack:
                #     if self.known_enemy_units.closer_than(10, u):
                #         # attack!
                #         # u.attack(random.choice(self.known_enemy_units.closer_than(10, u)))
                #         target=random.choice(self.known_enemy_units.closer_than(10, u))

                #     # if we can attack:
                #     elif self.known_enemy_structures.closer_than(10, u):
                #         # attack!
                #         # u.attack(random.choice(self.known_enemy_structures.closer_than(10, u)))
                #         target = random.choice(self.known_enemy_structures.closer_than(10, u))
                #     # any enemy units:
                #     elif self.known_enemy_units:
                #         # attack!
                #         # u.attack(random.choice(self.known_enemy_units))
                #         target = random.choice(self.known_enemy_units)
                #     # any enemy structures:
                #     elif self.known_enemy_structures:
                #         # attack!
                #         # u.attack(random.choice(self.known_enemy_structures))
                #         target = random.choice(self.known_enemy_structures)
                #     # if we can attack:
                #     elif self.enemy_start_locations:
                #         # attack!
                #         # u.attack(self.enemy_start_locations[0])
                #         target = self.enemy_start_locations[0]

                #     await self.do(u.attack(target)) 
            
            except Exception as e:
                print(e)
            

        #5: voidray flee (back to base)
        elif action == 5:
            if self.units(VOIDRAY).idle.amount > 0:
                for vr in self.units(UnitTypeId.VOIDRAY):
                    await self.do(vr.attack(self.start_location))


        map = np.zeros((self.game_info.map_size[0], self.game_info.map_size[1], 3), dtype=np.uint8)

        # draw the minerals:
        for mineral in self.state.mineral_field:
            pos = mineral.position
            c = [175, 255, 255]
            fraction = mineral.mineral_contents / 1800
            if mineral.is_visible:
                #print(mineral.mineral_contents)
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]
            else:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [20,75,50]  


        # draw the enemy start location:
        for enemy_start_location in self.enemy_start_locations:
            pos = enemy_start_location
            c = [0, 0, 255]
            map[math.ceil(pos.y)][math.ceil(pos.x)] = c

        # draw the enemy units:
        for enemy_unit in self.known_enemy_units:
            pos = enemy_unit.position
            c = [100, 0, 255]
            # get unit health fraction:
            fraction = enemy_unit.health / enemy_unit.health_max if enemy_unit.health_max > 0 else 0.0001
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]


        # draw the enemy structures:
        for enemy_structure in self.known_enemy_structures:
            pos = enemy_structure.position
            c = [0, 100, 255]
            # get structure health fraction:
            fraction = enemy_structure.health / enemy_structure.health_max if enemy_structure.health_max > 0 else 0.0001
            map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]

        # draw our structures:
        for our_structure in self.units.structure:
            # if it's a nexus:
            if our_structure.type_id == UnitTypeId.NEXUS:
                pos = our_structure.position
                c = [255, 255, 175]
                # get structure health fraction:
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]
            
            else:
                pos = our_structure.position
                c = [0, 255, 175]
                # get structure health fraction:
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]


        # draw the vespene geysers:
        for vespene in self.state.vespene_geyser:
            # draw these after buildings, since assimilators go over them. 
            # tried to denote some way that assimilator was on top, couldnt 
            # come up with anything. Tried by positions, but the positions arent identical. ie:
            # vesp position: (50.5, 63.5) 
            # bldg positions: [(64.369873046875, 58.982421875), (52.85693359375, 51.593505859375),...]
            pos = vespene.position
            c = [255, 175, 255]
            fraction = vespene.vespene_contents / 2250

            if vespene.is_visible:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]
            else:
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [50,20,75]

        # draw our units:
        for our_unit in self.units:
            # if it is a voidray:
            if our_unit.type_id == UnitTypeId.VOIDRAY:
                pos = our_unit.position
                c = [255, 75 , 75]
                # get health:
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]


            else:
                pos = our_unit.position
                c = [175, 255, 0]
                # get health:
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                map[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]

        # show map with opencv, resized to be larger:
        # horizontal flip:

        cv2.imshow('map',cv2.flip(cv2.resize(map, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), 0))
        cv2.waitKey(1)

        if SAVE_REPLAY:
            # save map image into "replays dir"
            cv2.imwrite(f"replays/{int(time.time())}-{iteration}.png", map)



        reward = 0

        try:
            attack_count = 0
            # iterate through our void rays:
            for voidray in self.units(UnitTypeId.VOIDRAY):
                # if voidray is attacking and is in range of enemy unit:
                if voidray.is_attacking and voidray.target_in_range:
                    if self.known_enemy_units.closer_than(8, voidray) or self.known_enemy_structures.closer_than(8, voidray):
                        # reward += 0.005 # original was 0.005, decent results, but let's 3x it. 
                        reward += 0.015  
                        attack_count += 1

        except Exception as e:
            print("reward",e)
            reward = 0

        
        if iteration % 100 == 0:
            print(f"Iter: {iteration}. RWD: {reward}. VR: {self.units(UnitTypeId.VOIDRAY).amount}")

        # write the file: 
        data = {"state": map, "reward": reward, "action": None, "done": False}  # empty action waiting for the next one!

        with open('state_rwd_action.pkl', 'wb') as f:
            pickle.dump(data, f)

        


result = run_game(  # run_game is a function that runs the game.
    maps.get("2000AtmospheresAIE"), # the map we are playing on
    [Bot(Race.Protoss, IncrediBot()), # runs our coded bot, protoss race, and we pass our bot object 
     Computer(Race.Zerg, Difficulty.Hard)], # runs a pre-made computer agent, zerg race, with a hard difficulty.
    realtime=False, # When set to True, the agent is limited in how long each step can take to process.
)


if str(result) == "Result.Victory":
    rwd = 500
else:
    rwd = -500

with open("results.txt","a") as f:
    f.write(f"{result}\n")


map = np.zeros((224, 224, 3), dtype=np.uint8)
observation = map
data = {"state": map, "reward": rwd, "action": None, "done": True}  # empty action waiting for the next one!
with open('state_rwd_action.pkl', 'wb') as f:
    pickle.dump(data, f)

cv2.destroyAllWindows()
cv2.waitKey(1)
time.sleep(3)
sys.exit()