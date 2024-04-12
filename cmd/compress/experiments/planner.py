from typing import List, Self
from models.selector import FilterSelector

class CGPPinPlanner(object):
    def __init__(self) -> None:
        self._plan: List[List[FilterSelector]] = []
        self._preliminary_plan: List[FilterSelector] = []

    def clone(self) -> Self:
        planner = CGPPinPlanner()
        planner._plan = self._plan[:]
        planner._preliminary_plan = self._preliminary_plan[:]
        return planner
        
    def add_mapping(self, sel: FilterSelector):
        self._preliminary_plan.append(sel)
    def next_mapping(self):
        self._plan.append(self._preliminary_plan[:])
        self._preliminary_plan.clear()
    def finish_mapping(self):
        if self._preliminary_plan:
            self._plan.append(self._preliminary_plan[:])
            self._preliminary_plan.clear()
    def get_plan(self):
        return iter(self._plan)
    