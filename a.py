import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot,Computer

class ascyBot(sc2.BotAI):
  async def on_step(self, iteration):
    print(iteration)
    await self.distribute_workers()


run_game(maps.get("AcropolisLE"), [Bot(Race.Protoss, ascyBot()), Computer(Race
.Zerg, Difficulty.Easy)], realtime=True)