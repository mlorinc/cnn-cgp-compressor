from typing import List, Self
import models.selector as selector

class CGPPinPlanner(object):
    def __init__(self) -> None:
        self._plan: List[List[selector.FilterSelector]] = []
        self._preliminary_plan: List[selector.FilterSelector] = []
        self.finished = False

    def clone(self) -> Self:
        planner = CGPPinPlanner()
        planner._plan = self._plan[:]
        planner._preliminary_plan = self._preliminary_plan[:]
        planner.finished = self.finished
        return planner
        
    def add_mapping(self, sel: selector.FilterSelector):
        assert not self.finished
        self._preliminary_plan.append(sel)
    def next_mapping(self):
        assert not self.finished
        self._plan.append(self._preliminary_plan[:])
        self._preliminary_plan.clear()
    def finish_mapping(self):
        self.finished = True
        if self._preliminary_plan:
            self._plan.append(self._preliminary_plan[:])
            self._preliminary_plan.clear()
    def get_plan(self):
        assert self.finished
        return iter(self._plan)
    