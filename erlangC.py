from math import exp, ceil, floor
class MyProgram:
    def __init__(self, transactions: float, aht: float, asa: float,
                 interval: int, shrinkage=0.0, occupancy = 1,
                 **kwargs):

        if transactions <= 0:
            raise ValueError("transactions can't be smaller or equals than 0")

        if aht <= 0:
            raise ValueError("aht can't be smaller or equals than 0")

        if asa <= 0:
            raise ValueError("asa can't be smaller or equals than 0")

        if interval <= 0:
            raise ValueError("interval can't be smaller or equals than 0")

        if shrinkage < 0 or shrinkage >= 1:
            raise ValueError("shrinkage must be between in the interval [0,1)")
            
        if occupancy < 0 or occupancy > 1:
            raise ValueError("occupancy must be between in the interval [0,1)")

        self.n_transactions = transactions
        self.aht = aht
        self.interval = interval
        self.asa = asa
        self.intensity = (self.n_transactions / self.interval) * self.aht
        self.shrinkage = shrinkage
        self.occupancy = occupancy

    def waiting_probability(self, positions: int, scale_positions: bool = False):

        if scale_positions:
            productive_positions = floor((1 - self.shrinkage) * positions)
        else:
            productive_positions = positions

        erlang_b_inverse = 1
        for position in range(1, productive_positions + 1):
            erlang_b_inverse = 1 + (erlang_b_inverse * position / self.intensity)

        erlang_b = 1 / erlang_b_inverse
        return productive_positions * erlang_b / (productive_positions - self.intensity * (1 - erlang_b))
        
    def service_level(self, positions: int, scale_positions: bool = True):
        if scale_positions:
            productive_positions = floor((1 - self.shrinkage) * positions )
        else:
            productive_positions = positions

        probability_wait = self.waiting_probability(productive_positions, scale_positions=False)
        exponential = exp(-(productive_positions - self.intensity) * (self.asa / self.aht))
        return max(0, 1 - (probability_wait * exponential))
    
 
    def achieved_occupancy(self, positions: int, scale_positions: bool = False):
        if scale_positions:
            productive_positions = floor((1 - self.shrinkage) * positions)
        else:
            productive_positions = positions

        return self.intensity / productive_positions
    
    def required_positions(self, service_level: float):
        if service_level < 0 or service_level > 1:
            raise ValueError("service_level must be between 0 and 1")

        positions = round(self.intensity + 1)
        achieved_service_level = self.service_level(positions, scale_positions=False)
        while achieved_service_level < service_level:
            positions += 1
            achieved_service_level = self.service_level(positions, scale_positions=False)

        achieved_occupancy = self.achieved_occupancy(positions, scale_positions=False)
        raw_positions = ceil(positions)

       
        if achieved_occupancy > self.occupancy:
            raw_positions = ceil(self.intensity / self.occupancy)
            achieved_occupancy = self.achieved_occupancy(raw_positions)
            achieved_service_level = self.service_level(raw_positions)


        positions = ceil(raw_positions / (1 - self.shrinkage))

        return {
                "positions": positions,
                "service_level": achieved_service_level}