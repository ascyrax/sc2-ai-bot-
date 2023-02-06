import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot,Computer
from sc2.constants import NEXUS, PROBE, PYLON
class ascyBot(sc2.BotAI):
  async def on_step(self, iteration):
    if iteration%1000==0:
      print(iteration)
    await self.distribute_workers()
    await self.build_workers(iteration)
    await self.build_pylons(iteration)
  
  async def build_workers(self,iteration):
    for nexus in self.units(NEXUS).ready.noqueue:
      if self.can_afford(PROBE):
        print(iteration, "worker built")
        await self.do(nexus.train(PROBE))

  async def build_pylons(self,iteration):
    if self.supply_left<5 and not self.already_pending(PYLON):
      nexuses = self.units(NEXUS).ready
      if nexuses.exists:
        if self.can_afford(PYLON):
          print(iteration, "pylon built")
          await self.build(PYLON, near=nexuses.first)

print(type(NEXUS),type(PROBE),type(PYLON))
run_game(maps.get("AcropolisLE"), [Bot(Race.Protoss, ascyBot()), Computer(Race
.Zerg, Difficulty.Easy)], realtime=True)