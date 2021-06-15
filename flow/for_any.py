# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
from common import com, TargetDomain
from flow.flow_item import FlowItem

class FlowDependNone(object):
    """
    machine_type に依存する制御を行うクラス

    flow =  FlowDependNone()
    """
    section_indices = ['00','01','02','03','04','05']
    targets = ['source', 'target']

    def __init__(self, f_target):
        self.f_target = f_target
        pass

    class _Item(FlowItem):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def items(self):
        """
        generate training items.
        a tuple of (machine_type, section_index, domain)
        """
        machine_type = '*' # ignore machine_type
        section_index = '*' # ignore section_index
        target = '*' # ignore target
        yield self._Item(machine_type, section_index, target)

    def domain(self, machine_type, section_index, target):
        machine_type = '*' # ignore machine_type
        section_index = '*' # ignore section_index
        target = '*' # ignore target
        return TargetDomain(machine_type, section_index, target)
