# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
from common import com, TargetDomain
from flow.flow_item import FlowItem

class FlowDependMachineSectionTarget(object):
    """
    machine_type, section_index, domain に依存する制御を行うクラス

    flow =  FlowDependMachineSectionTarget()
    """
    machine_types = ['ToyCar','ToyTrain','fan', 'gearbox','pump','slider','valve']
    section_indices_dev = ['00','01','02']  # for dev mode
    section_indices_eval = ['03','04','05'] # for eval mode
    targets = ['source', 'target']

    def __init__(self, f_target):
        self.f_target = f_target

    class _Item(FlowItem):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def items(self):
        """
        generate a tuple of (machine_type, section_index, domain)
        """
        usable_machine_types = com.param['limit']['usable_machine_types']
        for machine_type in self.machine_types:
            if usable_machine_types and machine_type not in usable_machine_types:
                continue # skip

            if com.mode: # dev mode
                section_indices = self.section_indices_dev
            else: # eval mode
                section_indices = self.section_indices_eval

            for section_index in section_indices:
                for target in self.targets:
                    yield self._Item(machine_type, section_index, target)

    def domain(self, machine_type, section_index, target):
        return TargetDomain(machine_type, section_index, target)
